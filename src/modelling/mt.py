"""
# Created by valler at 24/06/2024
Feature: 

"""


def _int32(x):
    # Get the 32 least significant bits.
    return int(0xFFFFFFFF & x)

class MT19937:

    def __init__(self, seed):
        # Initialize the index to 0
        self.index = 624
        self.mt = [0] * 624
        self.mt[0] = seed  # Initialize the initial state to the seed
        for i in range(1, 624):
            self.mt[i] = _int32(
                1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i)

    def extract_number(self):
        if self.index >= 624:
            self.twist()

        y = self.mt[self.index]

        # Right shift by 11 bits
        y = y ^ y >> 11
        # Shift y left by 7 and take the bitwise and of 2636928640
        y = y ^ y << 7 & 2636928640
        # Shift y left by 15 and take the bitwise and of y and 4022730752
        y = y ^ y << 15 & 4022730752
        # Right shift by 18 bits
        y = y ^ y >> 18

        self.index = self.index + 1

        return _int32(y)

    def twist(self):
        for i in range(624):
            # Get the most significant bit and add it to the less significant
            # bits of the next number
            y = _int32((self.mt[i] & 0x80000000) +
                       (self.mt[(i + 1) % 624] & 0x7fffffff))
            self.mt[i] = self.mt[(i + 397) % 624] ^ y >> 1

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x9908b0df
        self.index = 0


short_long_dict = {
    "klaji_k2": "Type of education",
    "optuki": "Received student financial aid",
    "ptoim1": "Main type of activity (TVM)",
    "ammattikoodi_k": "Code of occupation, 3-digit level",
    "tyke": "Number of unemployment months",
    "tyokk": "Months in employment",
    "akoko_k": "Size of household-dwelling unit",
    "asty": "Type of household-dwelling unit",
    "hape": "Tenure status of dwelling",
    "hulu": "Number of rooms (kitchen not included)",
    "taty": "Type of building",
    "vata": "Standard of equipment",
    "penulaika": "Age of youngest child in family",
    "peas": "Family status",
    "lkm_k": "Number of children in family",
    "a18lkm_k": "Number of all children aged under 18 in the family",
    "a7lkm_k": "Number of children aged under seven in the family",
    "a3lkm_k": "Number of children aged under three in the family",
    "pekoko_k": "Size of family",
    "pety": "Family type",
    "vela": "Old-age pension",
    "tkela": "Disability pension",
    "tyela": "Unemployment pension",
    "mela": "Special pension for farmers",
    "osela": "Part-time pension",
    "pela": "Survivor's pension",
    "yvela": "Individual early pension",
    "kturaha_k": "Disposable income",
    "velaty_k": "Debts in total",
    "lvar_k": "Taxable assets",
    "svatva_k": "Earned income total in state taxation",
    "palk_k": "Earned income",
    "tyotu_k": "Earned income",
    "tyrtuo_k": "Entrepreneurial income",
    "auto_k": "Car ownership"
}
