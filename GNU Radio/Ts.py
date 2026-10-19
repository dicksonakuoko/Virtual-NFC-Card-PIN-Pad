"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt
import logging

logger = logging.getLogger('grc')
logging.basicConfig(level=logging.DEBUG)

class blk(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(
        self,
        tx_fs=12.5e6,
        ask_percent=1.0,
        flagBits='00100100',
        commandCode=0b00000001,
        maskLength=0b00000000
    ):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Generate Command Data',   # will show up in GRC
            in_sig=[],
            out_sig=[np.complex64]
        )
        self.tx_fs = tx_fs
        self.ask_percent = ask_percent
        self.flagBits = str(flagBits)
        self.commandCode = commandCode
        self.maskLength = maskLength
        self.data_generated = False  # Add this attribute to track if data is generated
        self.command_data = None
        self.command_data_len = 0
        self.current_index = 0
        self.message_port_register_out(pmt.intern("tx_len"))

    def work(self, input_items, output_items):
        
        ## Generate Data
        if not self.data_generated:
            logger.debug("Flagbits: " + self.flagBits)
            flagBits = int(self.flagBits, 2)
            self.command_data = generateCommandData(
                self.tx_fs,
                self.ask_percent,
                flagBits,
                self.commandCode,
                self.maskLength
            )
            self.command_data_len = len(self.command_data)
            self.message_port_pub(
                pmt.intern("tx_len"),
                pmt.from_long(self.command_data_len)
            )
            self.data_generated = True
            logger.debug("DATA GENERATED")
        
        num_output_items = len(output_items[0])
        end_index = self.current_index + num_output_items
        
        ## Output data
        try:
            if end_index <= self.command_data_len:
                output_items[0][:] = self.command_data[
                    self.current_index:end_index
                ]
                self.current_index = end_index
            else:
                remaining_items = self.command_data_len - self.current_index
                output_items[0][:remaining_items] = self.command_data[
                    self.current_index:self.command_data_len
                ]
                output_items[0][remaining_items:] = self.command_data[
                    :num_output_items - remaining_items
                ]
                self.current_index = num_output_items - remaining_items
        except:
            self.data_generated = False
            self.command_data_len = 0
            self.current_index = 0
            remaining_items = 0
            output_items[0][:] = 0
            return 0

        return len(output_items[0])


def generateCommandData(tx_fs, ask_percent, flagBits, commandCode, maskLength):

    rf_reset_length = round((8160e-6)/(1/tx_fs))
    data_bytes = [flagBits, commandCode, maskLength]
    CRC = calculateCRC(data_bytes)

    ## Rearrange Bits for Transmission
    flagBits = bin(flagBits)[2:].zfill(8)
    commandCode = bin(commandCode)[2:].zfill(8)
    maskLength = bin(maskLength)[2:].zfill(8)

    rearranged_flagBits = ''
    for i in range(len(flagBits) - 1, 0, -2):
        bitpair = flagBits[i-1:i+1]
        rearranged_flagBits = rearranged_flagBits + bitpair

    rearranged_commandCode = ''
    for i in range(len(commandCode) - 1, 0, -2):
        bitpair = commandCode[i-1:i+1]
        rearranged_commandCode = rearranged_commandCode + bitpair

    rearranged_maskLength = ''
    for i in range(len(maskLength) - 1, 0, -2):
        bitpair = maskLength[i-1:i+1]
        rearranged_maskLength = rearranged_maskLength + bitpair

    rearranged_CRC = ''
    for i in range(len(CRC) - 1, 0, -2):
        bitpair = CRC[i-1:i+1]
        rearranged_CRC = rearranged_CRC + bitpair

    # logger.debug(rearranged_flagBits)
    # logger.debug(rearranged_commandCode)
    # logger.debug(rearranged_maskLength)
    # logger.debug(rearranged_CRC)
    # rearranged_flagBits = '10011000'
    # rearranged_commandCode = '01000000'
    # rearranged_maskLength = '00000000'
    # rearranged_CRC = '1001111110100000'

    command = (
        rearranged_flagBits
        + rearranged_commandCode
        + rearranged_maskLength
        + rearranged_CRC
    )

    command_data = modulateCommand(command, tx_fs, ask_percent)
    command_data = modulateSOF_EOF(command_data, tx_fs, ask_percent)
    command_data = modulateSlotData(
        command_data,
        tx_fs,
        ask_percent,
        flagBits
    )
    
    command_data = np.append(
        command_data,
        np.zeros(rf_reset_length,)
    )  # Add RF Reset after command
    
    command_data = addSineWave(command_data, tx_fs, 60000)

    return command_data


def calculateCRC(data_bytes):
    # Constants
    POLYNOMIAL = 0x8408  # x^16 + x^12 + x^5 + 1
    PRESET_VALUE = 0xFFFF
    NUMBER_OF_BYTES = len(data_bytes)
    
    current_crc_value = PRESET_VALUE
    
    for i in range(NUMBER_OF_BYTES):
        current_crc_value = current_crc_value ^ data_bytes[i]

        for j in range(8):
            if current_crc_value & 0x0001:
                current_crc_value = (
                    current_crc_value >> 1
                ) ^ POLYNOMIAL
            else:
                current_crc_value = current_crc_value >> 1

    inverted_crc_value = ~current_crc_value
    unsigned_integer = inverted_crc_value + (1 << 16)
    CRC = bin(unsigned_integer)[2:].zfill(16)

    return CRC
        

def modulateCommand(command, tx_fs, ask_percent):
    
    pair_samples = round((75.52e-6)/(1/tx_fs))
    low_percent = 1 - ask_percent
    low_samples = round((9.44e-06)/(1/tx_fs))
    low_data = np.ones(low_samples) * low_percent
    
    data = []

    for i in range(0, len(command), 2):
        pair = command[i:i+2]
        pair_data = np.ones(pair_samples)

        match pair:
            case "00":
                high_samples = round((9.44e-06)/(1/tx_fs))
                pair_data[
                    high_samples:high_samples+low_samples
                ] = low_data

            case "01":
                high_samples = round((28.32e-06)/(1/tx_fs))
                pair_data[
                    high_samples:high_samples+low_samples
                ] = low_data

            case "10":
                high_samples = round((47.2e-06)/(1/tx_fs))
                pair_data[
                    high_samples:high_samples+low_samples
                ] = low_data

            case "11":
                high_samples = round((66.08e-06)/(1/tx_fs))
                pair_data[
                    high_samples:high_samples+low_samples
                ] = low_data

        data = np.append(data, pair_data)

    return data


def modulateSOF_EOF(data, tx_fs, ask_percent):
    SOF_length = round((75.52e-06)/(1/tx_fs))
    EOF_length = round(SOF_length/2)
    
    low_percent = 1-ask_percent
    low_samples = round((9.44e-06)/(1/tx_fs))
    low_data = np.ones(low_samples) * low_percent

    SOF = np.ones(SOF_length)
    SOF[:low_samples] = low_data

    high_samples = round((47.2e-06)/(1/tx_fs))
    SOF[
        high_samples:high_samples+low_samples
    ] = low_data

    EOF = np.ones(EOF_length)

    high_samples = round((18.88e-06)/(1/tx_fs))
    EOF[
        high_samples:high_samples + low_samples
    ] = low_data

    data = np.append(SOF, data)
    data = np.append(data, EOF)

    return data


def modulateSlotData(data, tx_fs, ask_percent, flagBits):
    slot_flag = flagBits[2]
    rate_flag = flagBits[6]
    inv_flag = flagBits[5]
    
    low_percent = 1-ask_percent
    low_samples = round((9.44e-06)/(1/tx_fs))
    low_data = np.ones(low_samples) * low_percent

    EOF_length = round((37.76e-06)/(1/tx_fs))
    EOF = np.ones(EOF_length)

    high_samples = round((18.88e-06)/(1/tx_fs))
    EOF[
        high_samples:high_samples + low_samples
    ] = low_data

    if rate_flag == '1':
        slot_length = round(8160e-06/(1/tx_fs))
    else:
        slot_length = round(20332e-06/(1/tx_fs))

    slot = np.ones(slot_length)
    data = np.append(slot, data)

    if slot_flag == '0' and inv_flag == '1':
        slot = np.append(slot, EOF)

        for i in range(16):
            data = np.append(data, slot)
    else:
        data = np.append(
            data,
            np.ones(slot_length)
        )

    return data


def addSineWave(data, tx_fs, sinewave_fs):
    data_length = len(data)
    duration = data_length / tx_fs

    t = np.linspace(
        0,
        duration,
        data_length,
        endpoint=False,
        dtype=np.complex64
    )  # Time array

    # Generate the complex sine wave
    sine_wave = np.exp(
        2j * np.pi * sinewave_fs * t
    ).astype(np.complex64)

    data = sine_wave * data

    return data
