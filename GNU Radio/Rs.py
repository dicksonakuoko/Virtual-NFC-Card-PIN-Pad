"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__ will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt
import logging
import os
from pathlib import Path
from scipy.stats.stats import pearsonr

logger = logging.getLogger('grc')
logging.basicConfig(level=logging.DEBUG)


class SaveNumpyArrayBlock(gr.sync_block):

    def __init__(
        self,
        rx_fs=2e6,
        tx_fs=12.5e6,
        flagBits='00100100',
        signals_requested=200,
        card=1,
        button=0
    ):
        gr.sync_block.__init__(
            self,
            name='Save Card Button Signals',
            in_sig=[np.complex64],
            out_sig=[]
        )

        ## Make variables persistent with each loop
        self.data = []
        self.rx_fs = rx_fs
        self.tx_fs = tx_fs
        self.flagBits = str(flagBits)
        self.signals_found = 0
        self.signals_requested = signals_requested

        # Disable the pause at 50 responses
        self.responses_per_placement = self.signals_requested

        # Pause collection after each group
        self.collection_paused = False

        self.tx_len = 0
        self.card = int(card)
        self.button = int(button)

        if self.card < 1 or self.card > 40:
            raise ValueError(
                "card must be between 1 and 40"
            )

        if self.button < 0 or self.button > 11:
            raise ValueError(
                "button must be between 0 and 11"
            )

        if len(self.flagBits) != 8:
            raise ValueError(
                "flagBits must contain exactly 8 bits"
            )

        if any(bit not in ('0', '1') for bit in self.flagBits):
            raise ValueError(
                "flagBits must contain only 0 and 1"
            )

        self.message_port_register_in(
            pmt.intern("tx_len")
        )

        self.set_msg_handler(
            pmt.intern("tx_len"),
            self.handle_msg
        )

        logger.debug(
            "Configured collection: Card "
            + str(self.card)
            + ", Button "
            + str(self.button)
            + ", FlagBits "
            + self.flagBits
        )


    # Get tx_len from generate command data block
    def handle_msg(self, msg):
        self.tx_len = pmt.to_long(msg)


    def work(self, input_items, output_items):

        # Pause after each group of responses.
        # Reposition the induced coil, stop the flowgraph,
        # and restart it to continue collection.
        if self.collection_paused:
            return 0

        # Use manually selected flagBits from GRC
        flagBits = self.flagBits

        # Change variables based on data-rate and subcarrier combination
        match flagBits[6:8]:

            case '00':
                data_directory = Path(
                    '/home/cse/pinpad/myenv/GNURadio_1/'
                    'Correlation_Responses/'
                    'OneSubcarrierLowDataRate'
                )

                SOF_time = (151.04e-06) * 4
                bit_time = (37.76e-06) * 4

            case '01':
                data_directory = Path(
                    '/home/cse/pinpad/myenv/GNURadio_1/'
                    'Correlation_Responses/'
                    'TwoSubcarriersLowDataRate'
                )

                SOF_time = (149.85e-06) * 4
                bit_time = (37.46e-06) * 4

            case '10':
                data_directory = Path(
                    '/home/cse/pinpad/myenv/GNURadio_1/'
                    'Correlation_Responses/'
                    'OneSubcarrierHighDataRate'
                )

                SOF_time = 151.04e-06
                bit_time = 37.76e-06

            case '11':
                data_directory = Path(
                    '/home/cse/pinpad/myenv/GNURadio_1/'
                    'Correlation_Responses/'
                    'TwoSubcarriersHighDataRate'
                )

                SOF_time = 149.85e-06
                bit_time = 37.46e-06

        # Create a separate folder for each card
        card_directory = (
            data_directory
            / ('card' + str(self.card))
        )

        os.makedirs(
            card_directory,
            exist_ok=True
        )

        # Save one file for each button position
        button_name = (
            'button'
            + str(self.button)
            + '.npy'
        )

        self.card_sig_path = (
            card_directory
            / button_name
        )

        # Correlation signal remains inside the mode directory
        self.corr_sig_path = (
            data_directory
            / 'corr_signal.npy'
        )

        SOF_length = round(
            SOF_time / (1 / self.rx_fs)
        )

        bit_length = round(
            bit_time / (1 / self.rx_fs)
        )

        response_bits = 8

        response_length = (
            SOF_length
            + bit_length * response_bits
        )

        # Create the button response file if it does not exist
        if not self.card_sig_path.is_file():
            np.save(
                self.card_sig_path,
                np.zeros(
                    (0, response_length),
                    dtype=np.complex64
                )
            )

        # Load the known response used for correlation filtering
        self.corr_signal = np.load(
            self.corr_sig_path,
            allow_pickle=True
        )

        # Process after collecting the required buffer size
        if len(self.data) >= 2000000:

            prev_cards = np.load(
                self.card_sig_path,
                allow_pickle=True
            )

            self.signals_found = (
                prev_cards.shape[0]
            )

            buffer = np.array(
                self.data,
                dtype=np.complex64
            )

            try:
                card_signals = findCardSignals(
                    buffer,
                    self.rx_fs,
                    self.tx_fs,
                    flagBits,
                    self.tx_len
                )

                card_signals = normalize_data(
                    self,
                    card_signals
                )

            except Exception as error:
                logger.debug(
                    "Signal extraction error: "
                    + str(error)
                )

                self.data = []
                return 0

            ## Remove signals that are not similar to the correlation signal
            row_to_remove = []

            for i in range(
                card_signals.shape[0]
            ):

                diff = pearsonr(
                    np.abs(self.corr_signal),
                    np.abs(card_signals[i])
                )[0]

                if diff < 0.8:
                    row_to_remove.append(i)

            card_signals = np.delete(
                card_signals,
                row_to_remove,
                axis=0
            )

            ## Determine total number of responses still required
            signals_remaining = (
                self.signals_requested
                - self.signals_found
            )

            ## Determine progress within the current response batch
            responses_in_current_batch = (
                self.signals_found
                % self.responses_per_placement
            )

            ## Determine how many more responses can be collected
            ## before pausing
            responses_until_pause = (
                self.responses_per_placement
                - responses_in_current_batch
            )

            ## Limit collection by:
            ## 1. Total responses remaining
            ## 2. Responses remaining in the current batch
            ## 3. Number of valid responses currently detected
            signals_remaining = min(
                signals_remaining,
                responses_until_pause,
                card_signals.shape[0]
            )

            ## Add and save newly found responses
            card_signals = np.vstack(
                (
                    prev_cards,
                    card_signals[
                        :signals_remaining
                    ]
                )
            )

            self.signals_found = (
                card_signals.shape[0]
            )

            logger.debug(
                "Card "
                + str(self.card)
                + ", Button "
                + str(self.button)
                + ", Signals Found: "
                + str(self.signals_found)
                + "/"
                + str(self.signals_requested)
            )

            np.save(
                self.card_sig_path,
                card_signals
            )

            ## Pause after collecting each group
            if (
                self.signals_found > 0
                and self.signals_found
                < self.signals_requested
                and self.signals_found
                % self.responses_per_placement == 0
            ):

                logger.debug(
                    "=================================================="
                )

                logger.debug(
                    "PAUSED: Card "
                    + str(self.card)
                    + ", Button "
                    + str(self.button)
                    + " has collected "
                    + str(self.signals_found)
                    + "/"
                    + str(self.signals_requested)
                    + " responses."
                )

                logger.debug(
                    "Reposition the induced button coil."
                )

                logger.debug(
                    "Then stop and restart the flowgraph "
                    "to continue collecting responses."
                )

                logger.debug(
                    "Current data file: "
                    + str(self.card_sig_path)
                )

                logger.debug(
                    "=================================================="
                )

                self.collection_paused = True
                self.data = []

                return 0

            ## Check whether collection is complete
            if (
                self.signals_found
                >= self.signals_requested
            ):

                card_signals = normalize_data(
                    self,
                    card_signals
                )

                np.save(
                    self.card_sig_path,
                    card_signals
                )

                logger.debug(
                    "------- CARD "
                    + str(self.card)
                    + ", BUTTON "
                    + str(self.button)
                    + ", FLAGBITS "
                    + str(self.flagBits)
                    + " COMPLETED -------"
                )

                logger.debug(
                    "Total responses saved: "
                    + str(self.signals_found)
                )

                logger.debug(
                    "Saved file: "
                    + str(self.card_sig_path)
                )

                self.data = []

            else:
                self.data = []

            return 0

        ## Collect incoming samples
        elif len(self.data) <= 2000000:

            self.data.extend(
                input_items[0]
            )

            return len(
                input_items[0]
            )

        return 0


def findCardSignals(
    data,
    rx_fs,
    tx_fs,
    flagBits,
    tx_len
):
    abs_data = np.abs(data)

    ## Get the correct timings of received data
    rate_flag = flagBits[6]

    if rate_flag == '1':
        slot_length = round(
            8160e-06 / (1 / rx_fs)
        )
    else:
        slot_length = round(
            20332e-06 / (1 / rx_fs)
        )

    rf_reset_length = round(
        8160e-06 / (1 / rx_fs)
    )

    command_time = (
        tx_len / tx_fs
    )

    command_length = round(
        command_time / (1 / rx_fs)
        + 2.5e-04 / (1 / rx_fs)
    )

    N = 50
    rho = 15

    match flagBits[6:8]:

        case '00':
            SOF_time = (
                151.04e-06
            ) * 4

            SOF_low_time = (
                56.64e-06
            ) * 4

            bit_time = (
                37.76e-06
            ) * 4

        case '01':
            SOF_time = (
                149.85e-06
            ) * 4

            SOF_low_time = 0

            bit_time = (
                37.46e-06
            ) * 4

        case '10':
            SOF_time = 151.04e-06
            SOF_low_time = 56.64e-06
            bit_time = 37.76e-06

        case '11':
            SOF_time = 149.85e-06
            SOF_low_time = 0
            bit_time = 37.46e-06

    SOF_length = round(
        SOF_time / (1 / rx_fs)
    )

    SOF_low_length = round(
        SOF_low_time / (1 / rx_fs)
    )

    bit_length = round(
        bit_time / (1 / rx_fs)
    )

    response_bits = 8

    response_length = (
        SOF_length
        + bit_length * response_bits
    )

    tx_start = 0

    card_signals = (
        [None] * response_length
    )

    ## Find when the received signal is low
    start = findRFreset(data)
    data = data[start:]

    ## Loop through the buffer to find card responses
    while (
        tx_start
        + 2 * command_length
        < len(abs_data)
    ):

        abs_data = np.abs(data)

        # Find the start of command transmission
        for i in range(
            0,
            len(abs_data),
            100
        ):
            window_size = 100

            if (
                i + window_size
                > len(abs_data) - 1
            ):
                window_size = (
                    len(abs_data)
                    - i
                    - 1
                )

            delta = np.abs(
                abs_data[
                    i + window_size
                ]
                - abs_data[i]
            )

            if delta > 0.1:
                tx_start = i
                break

            tx_start = None

        if tx_start is None:
            return card_signals[1:]

        offset = (
            tx_start
            + command_length
            - slot_length
            - rf_reset_length
        )

        # Find the start of the card response
        response_start = findResponseStart(
            data[offset:],
            N,
            rho
        )

        offset = (
            offset
            + response_start
            - SOF_low_length
        )

        # Extract the card response
        card_signal = data[
            offset:
            offset + response_length
        ]

        card_signals = np.vstack(
            (
                card_signals,
                card_signal
            )
        )

        # Find the next transmission interval
        data = data[
            tx_start
            + command_length
            - rf_reset_length:
        ]

        start = findRFreset(data)
        data = data[start:]

    return card_signals[1:]


def findResponseStart(
    data,
    N,
    rho
):
    data = np.abs(data)
    delta = np.abs(np.diff(data))

    midPoint = int(
        np.floor(N / 2) + 1
    )

    deltaSegment = delta[
        midPoint:N
    ]

    sigma = (
        rho
        * np.mean(deltaSegment)
    )

    start = np.where(
        delta > sigma
    )[0][0]

    return start


def findRFreset(data):

    window_size = 1000

    for i in range(
        0,
        len(data),
        window_size
    ):
        avg = np.mean(
            abs(
                data[
                    i:
                    i + window_size
                ]
            )
        )

        if avg <= 0.001:
            return i


## Uses a sliding window and mean to determine similarity
def differences(
    arr1,
    arr2
):
    windows_size = 10
    diffs = []

    for i in range(
        0,
        len(arr1) - windows_size,
        windows_size
    ):
        arr1_mean = np.mean(
            np.abs(
                arr1[
                    i:
                    i + windows_size
                ]
            )
        )

        arr2_mean = np.mean(
            np.abs(
                arr2[
                    i:
                    i + windows_size
                ]
            )
        )

        diff = np.abs(
            arr1_mean
            - arr2_mean
        )

        diffs = np.append(
            diffs,
            diff
        )

    return np.mean(diffs)


## Make each input response have the same mean as the correlation signal
def normalize_data(
    self,
    data
):
    data = np.asarray(
        data,
        np.complex_
    )

    overall_mean_magnitude = np.mean(
        np.abs(self.corr_signal)
    )

    row_means = np.mean(
        np.abs(data),
        axis=1
    )

    normalized_data = np.empty_like(
        data,
        dtype=np.complex_
    )

    for i in range(
        data.shape[0]
    ):
        magnitudes = np.abs(
            data[i]
        )

        phases = np.angle(
            data[i]
        )

        scale_factor = (
            overall_mean_magnitude
            / row_means[i]
        )

        new_magnitudes = (
            magnitudes
            * scale_factor
        )

        normalized_data[i] = (
            new_magnitudes
            * np.exp(
                1j * phases
            )
        )

    return normalized_data
