# -*- encoding: utf-8 -*-

import numpy as np
import nnhealpix as nnh

def test_dgrade():
    ref41 = np.array([
        0, 4, 5, 12, 13, 14, 24, 25, 26, 27, 41, 42, 43, 57, 58,
        74, 1, 6, 7, 15, 16, 17, 28, 29, 30, 31, 45, 46, 47, 61,
        62, 78, 2, 8, 9, 18, 19, 20, 32, 33, 34, 35, 49, 50, 51,
        65, 66, 82, 3, 10, 11, 21, 22, 23, 36, 37, 38, 39, 53, 54,
        55, 69, 70, 86, 40, 56, 71, 72, 73, 87, 88, 89, 102, 103, 104,
        105, 119, 120, 135, 136, 44, 59, 60, 75, 76, 77, 90, 91, 92, 93,
        107, 108, 109, 123, 124, 140, 48, 63, 64, 79, 80, 81, 94, 95, 96,
        97, 111, 112, 113, 127, 128, 144, 52, 67, 68, 83, 84, 85, 98, 99,
        100, 101, 115, 116, 117, 131, 132, 148, 106, 121, 122, 137, 138, 139, 152,
        153, 154, 155, 168, 169, 170, 180, 181, 188, 110, 125, 126, 141, 142, 143,
        156, 157, 158, 159, 171, 172, 173, 182, 183, 189, 114, 129, 130, 145, 146,
        147, 160, 161, 162, 163, 174, 175, 176, 184, 185, 190, 118, 133, 134, 149,
        150, 151, 164, 165, 166, 167, 177, 178, 179, 186, 187, 191,
    ], dtype='int')

    ref42 = np.array([
        0, 4, 5, 13, 1, 6, 7, 16, 2, 8, 9, 19, 3, 10, 11,
        22, 12, 24, 25, 41, 14, 26, 27, 43, 15, 28, 29, 45, 17, 30,
        31, 47, 18, 32, 33, 49, 20, 34, 35, 51, 21, 36, 37, 53, 23,
        38, 39, 55, 40, 56, 71, 72, 42, 57, 58, 74, 44, 59, 60, 76,
        46, 61, 62, 78, 48, 63, 64, 80, 50, 65, 66, 82, 52, 67, 68,
        84, 54, 69, 70, 86, 73, 88, 89, 105, 75, 90, 91, 107, 77, 92,
        93, 109, 79, 94, 95, 111, 81, 96, 97, 113, 83, 98, 99, 115, 85,
        100, 101, 117, 87, 102, 103, 119, 104, 120, 135, 136, 106, 121, 122, 138,
        108, 123, 124, 140, 110, 125, 126, 142, 112, 127, 128, 144, 114, 129, 130,
        146, 116, 131, 132, 148, 118, 133, 134, 150, 137, 152, 153, 168, 139, 154,
        155, 170, 141, 156, 157, 171, 143, 158, 159, 173, 145, 160, 161, 174, 147,
        162, 163, 176, 149, 164, 165, 177, 151, 166, 167, 179, 169, 180, 181, 188,
        172, 182, 183, 189, 175, 184, 185, 190, 178, 186, 187, 191,
    ], dtype='int')

    ref81 = np.array([
        0, 4, 5, 12, 13, 14, 24, 25, 26, 27, 40, 41, 42, 43, 44,
        60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 89, 90, 112, 113,
        114, 115, 116, 117, 118, 119, 145, 146, 147, 148, 149, 150, 151, 177, 178,
        179, 180, 181, 182, 210, 211, 212, 213, 214, 242, 243, 244, 245, 275, 276,
        277, 307, 308, 340, 1, 6, 7, 15, 16, 17, 28, 29, 30, 31, 45,
        46, 47, 48, 49, 66, 67, 68, 69, 70, 71, 91, 92, 93, 94, 95,
        96, 97, 120, 121, 122, 123, 124, 125, 126, 127, 153, 154, 155, 156, 157,
        158, 159, 185, 186, 187, 188, 189, 190, 218, 219, 220, 221, 222, 250, 251,
        252, 253, 283, 284, 285, 315, 316, 348, 2, 8, 9, 18, 19, 20, 32,
        33, 34, 35, 50, 51, 52, 53, 54, 72, 73, 74, 75, 76, 77, 98,
        99, 100, 101, 102, 103, 104, 128, 129, 130, 131, 132, 133, 134, 135, 161,
        162, 163, 164, 165, 166, 167, 193, 194, 195, 196, 197, 198, 226, 227, 228,
        229, 230, 258, 259, 260, 261, 291, 292, 293, 323, 324, 356, 3, 10, 11,
        21, 22, 23, 36, 37, 38, 39, 55, 56, 57, 58, 59, 78, 79, 80,
        81, 82, 83, 105, 106, 107, 108, 109, 110, 111, 136, 137, 138, 139, 140,
        141, 142, 143, 169, 170, 171, 172, 173, 174, 175, 201, 202, 203, 204, 205,
        206, 234, 235, 236, 237, 238, 266, 267, 268, 269, 299, 300, 301, 331, 332,
        364, 144, 176, 207, 208, 209, 239, 240, 241, 270, 271, 272, 273, 274, 302,
        303, 304, 305, 306, 333, 334, 335, 336, 337, 338, 339, 365, 366, 367, 368,
        369, 370, 371, 396, 397, 398, 399, 400, 401, 402, 403, 429, 430, 431, 432,
        433, 434, 461, 462, 463, 464, 465, 466, 494, 495, 496, 497, 526, 527, 528,
        529, 559, 560, 591, 592, 152, 183, 184, 215, 216, 217, 246, 247, 248, 249,
        278, 279, 280, 281, 282, 309, 310, 311, 312, 313, 314, 341, 342, 343, 344,
        345, 346, 347, 372, 373, 374, 375, 376, 377, 378, 379, 405, 406, 407, 408,
        409, 410, 411, 437, 438, 439, 440, 441, 442, 470, 471, 472, 473, 474, 502,
        503, 504, 505, 535, 536, 537, 567, 568, 600, 160, 191, 192, 223, 224, 225,
        254, 255, 256, 257, 286, 287, 288, 289, 290, 317, 318, 319, 320, 321, 322,
        349, 350, 351, 352, 353, 354, 355, 380, 381, 382, 383, 384, 385, 386, 387,
        413, 414, 415, 416, 417, 418, 419, 445, 446, 447, 448, 449, 450, 478, 479,
        480, 481, 482, 510, 511, 512, 513, 543, 544, 545, 575, 576, 608, 168, 199,
        200, 231, 232, 233, 262, 263, 264, 265, 294, 295, 296, 297, 298, 325, 326,
        327, 328, 329, 330, 357, 358, 359, 360, 361, 362, 363, 388, 389, 390, 391,
        392, 393, 394, 395, 421, 422, 423, 424, 425, 426, 427, 453, 454, 455, 456,
        457, 458, 486, 487, 488, 489, 490, 518, 519, 520, 521, 551, 552, 553, 583,
        584, 616, 404, 435, 436, 467, 468, 469, 498, 499, 500, 501, 530, 531, 532,
        533, 534, 561, 562, 563, 564, 565, 566, 593, 594, 595, 596, 597, 598, 599,
        624, 625, 626, 627, 628, 629, 630, 631, 656, 657, 658, 659, 660, 661, 662,
        684, 685, 686, 687, 688, 689, 708, 709, 710, 711, 712, 728, 729, 730, 731,
        744, 745, 746, 756, 757, 764, 412, 443, 444, 475, 476, 477, 506, 507, 508,
        509, 538, 539, 540, 541, 542, 569, 570, 571, 572, 573, 574, 601, 602, 603,
        604, 605, 606, 607, 632, 633, 634, 635, 636, 637, 638, 639, 663, 664, 665,
        666, 667, 668, 669, 690, 691, 692, 693, 694, 695, 713, 714, 715, 716, 717,
        732, 733, 734, 735, 747, 748, 749, 758, 759, 765, 420, 451, 452, 483, 484,
        485, 514, 515, 516, 517, 546, 547, 548, 549, 550, 577, 578, 579, 580, 581,
        582, 609, 610, 611, 612, 613, 614, 615, 640, 641, 642, 643, 644, 645, 646,
        647, 670, 671, 672, 673, 674, 675, 676, 696, 697, 698, 699, 700, 701, 718,
        719, 720, 721, 722, 736, 737, 738, 739, 750, 751, 752, 760, 761, 766, 428,
        459, 460, 491, 492, 493, 522, 523, 524, 525, 554, 555, 556, 557, 558, 585,
        586, 587, 588, 589, 590, 617, 618, 619, 620, 621, 622, 623, 648, 649, 650,
        651, 652, 653, 654, 655, 677, 678, 679, 680, 681, 682, 683, 702, 703, 704,
        705, 706, 707, 723, 724, 725, 726, 727, 740, 741, 742, 743, 753, 754, 755,
        762, 763, 767,
    ], dtype='int')
    
    ref82 = np.array([
        0, 4, 5, 12, 13, 14, 24, 25, 26, 27, 41, 42, 43, 62, 63,
        87, 1, 6, 7, 15, 16, 17, 28, 29, 30, 31, 46, 47, 48, 68,
        69, 94, 2, 8, 9, 18, 19, 20, 32, 33, 34, 35, 51, 52, 53,
        74, 75, 101, 3, 10, 11, 21, 22, 23, 36, 37, 38, 39, 56, 57,
        58, 80, 81, 108, 40, 60, 61, 84, 85, 86, 112, 113, 114, 115, 145,
        146, 147, 177, 178, 210, 44, 64, 65, 88, 89, 90, 116, 117, 118, 119,
        149, 150, 151, 181, 182, 214, 45, 66, 67, 91, 92, 93, 120, 121, 122,
        123, 153, 154, 155, 185, 186, 218, 49, 70, 71, 95, 96, 97, 124, 125,
        126, 127, 157, 158, 159, 189, 190, 222, 50, 72, 73, 98, 99, 100, 128,
        129, 130, 131, 161, 162, 163, 193, 194, 226, 54, 76, 77, 102, 103, 104,
        132, 133, 134, 135, 165, 166, 167, 197, 198, 230, 55, 78, 79, 105, 106,
        107, 136, 137, 138, 139, 169, 170, 171, 201, 202, 234, 59, 82, 83, 109,
        110, 111, 140, 141, 142, 143, 173, 174, 175, 205, 206, 238, 144, 176, 207,
        208, 209, 239, 240, 241, 270, 271, 272, 273, 303, 304, 335, 336, 148, 179,
        180, 211, 212, 213, 242, 243, 244, 245, 275, 276, 277, 307, 308, 340, 152,
        183, 184, 215, 216, 217, 246, 247, 248, 249, 279, 280, 281, 311, 312, 344,
        156, 187, 188, 219, 220, 221, 250, 251, 252, 253, 283, 284, 285, 315, 316,
        348, 160, 191, 192, 223, 224, 225, 254, 255, 256, 257, 287, 288, 289, 319,
        320, 352, 164, 195, 196, 227, 228, 229, 258, 259, 260, 261, 291, 292, 293,
        323, 324, 356, 168, 199, 200, 231, 232, 233, 262, 263, 264, 265, 295, 296,
        297, 327, 328, 360, 172, 203, 204, 235, 236, 237, 266, 267, 268, 269, 299,
        300, 301, 331, 332, 364, 274, 305, 306, 337, 338, 339, 368, 369, 370, 371,
        401, 402, 403, 433, 434, 466, 278, 309, 310, 341, 342, 343, 372, 373, 374,
        375, 405, 406, 407, 437, 438, 470, 282, 313, 314, 345, 346, 347, 376, 377,
        378, 379, 409, 410, 411, 441, 442, 474, 286, 317, 318, 349, 350, 351, 380,
        381, 382, 383, 413, 414, 415, 445, 446, 478, 290, 321, 322, 353, 354, 355,
        384, 385, 386, 387, 417, 418, 419, 449, 450, 482, 294, 325, 326, 357, 358,
        359, 388, 389, 390, 391, 421, 422, 423, 453, 454, 486, 298, 329, 330, 361,
        362, 363, 392, 393, 394, 395, 425, 426, 427, 457, 458, 490, 302, 333, 334,
        365, 366, 367, 396, 397, 398, 399, 429, 430, 431, 461, 462, 494, 400, 432,
        463, 464, 465, 495, 496, 497, 526, 527, 528, 529, 559, 560, 591, 592, 404,
        435, 436, 467, 468, 469, 498, 499, 500, 501, 531, 532, 533, 563, 564, 596,
        408, 439, 440, 471, 472, 473, 502, 503, 504, 505, 535, 536, 537, 567, 568,
        600, 412, 443, 444, 475, 476, 477, 506, 507, 508, 509, 539, 540, 541, 571,
        572, 604, 416, 447, 448, 479, 480, 481, 510, 511, 512, 513, 543, 544, 545,
        575, 576, 608, 420, 451, 452, 483, 484, 485, 514, 515, 516, 517, 547, 548,
        549, 579, 580, 612, 424, 455, 456, 487, 488, 489, 518, 519, 520, 521, 551,
        552, 553, 583, 584, 616, 428, 459, 460, 491, 492, 493, 522, 523, 524, 525,
        555, 556, 557, 587, 588, 620, 530, 561, 562, 593, 594, 595, 624, 625, 626,
        627, 656, 657, 658, 684, 685, 708, 534, 565, 566, 597, 598, 599, 628, 629,
        630, 631, 660, 661, 662, 688, 689, 712, 538, 569, 570, 601, 602, 603, 632,
        633, 634, 635, 663, 664, 665, 690, 691, 713, 542, 573, 574, 605, 606, 607,
        636, 637, 638, 639, 667, 668, 669, 694, 695, 717, 546, 577, 578, 609, 610,
        611, 640, 641, 642, 643, 670, 671, 672, 696, 697, 718, 550, 581, 582, 613,
        614, 615, 644, 645, 646, 647, 674, 675, 676, 700, 701, 722, 554, 585, 586,
        617, 618, 619, 648, 649, 650, 651, 677, 678, 679, 702, 703, 723, 558, 589,
        590, 621, 622, 623, 652, 653, 654, 655, 681, 682, 683, 706, 707, 727, 659,
        686, 687, 709, 710, 711, 728, 729, 730, 731, 744, 745, 746, 756, 757, 764,
        666, 692, 693, 714, 715, 716, 732, 733, 734, 735, 747, 748, 749, 758, 759,
        765, 673, 698, 699, 719, 720, 721, 736, 737, 738, 739, 750, 751, 752, 760,
        761, 766, 680, 704, 705, 724, 725, 726, 740, 741, 742, 743, 753, 754, 755,
        762, 763, 767,
    ], dtype='int')
    
    ref84 = np.array([
        0, 4, 5, 13, 1, 6, 7, 16, 2, 8, 9, 19, 3, 10, 11,
        22, 12, 24, 25, 41, 14, 26, 27, 43, 15, 28, 29, 46, 17, 30,
        31, 48, 18, 32, 33, 51, 20, 34, 35, 53, 21, 36, 37, 56, 23,
        38, 39, 58, 40, 60, 61, 85, 42, 62, 63, 87, 44, 64, 65, 89,
        45, 66, 67, 92, 47, 68, 69, 94, 49, 70, 71, 96, 50, 72, 73,
        99, 52, 74, 75, 101, 54, 76, 77, 103, 55, 78, 79, 106, 57, 80,
        81, 108, 59, 82, 83, 110, 84, 112, 113, 145, 86, 114, 115, 147, 88,
        116, 117, 149, 90, 118, 119, 151, 91, 120, 121, 153, 93, 122, 123, 155,
        95, 124, 125, 157, 97, 126, 127, 159, 98, 128, 129, 161, 100, 130, 131,
        163, 102, 132, 133, 165, 104, 134, 135, 167, 105, 136, 137, 169, 107, 138,
        139, 171, 109, 140, 141, 173, 111, 142, 143, 175, 144, 176, 207, 208, 146,
        177, 178, 210, 148, 179, 180, 212, 150, 181, 182, 214, 152, 183, 184, 216,
        154, 185, 186, 218, 156, 187, 188, 220, 158, 189, 190, 222, 160, 191, 192,
        224, 162, 193, 194, 226, 164, 195, 196, 228, 166, 197, 198, 230, 168, 199,
        200, 232, 170, 201, 202, 234, 172, 203, 204, 236, 174, 205, 206, 238, 209,
        240, 241, 273, 211, 242, 243, 275, 213, 244, 245, 277, 215, 246, 247, 279,
        217, 248, 249, 281, 219, 250, 251, 283, 221, 252, 253, 285, 223, 254, 255,
        287, 225, 256, 257, 289, 227, 258, 259, 291, 229, 260, 261, 293, 231, 262,
        263, 295, 233, 264, 265, 297, 235, 266, 267, 299, 237, 268, 269, 301, 239,
        270, 271, 303, 272, 304, 335, 336, 274, 305, 306, 338, 276, 307, 308, 340,
        278, 309, 310, 342, 280, 311, 312, 344, 282, 313, 314, 346, 284, 315, 316,
        348, 286, 317, 318, 350, 288, 319, 320, 352, 290, 321, 322, 354, 292, 323,
        324, 356, 294, 325, 326, 358, 296, 327, 328, 360, 298, 329, 330, 362, 300,
        331, 332, 364, 302, 333, 334, 366, 337, 368, 369, 401, 339, 370, 371, 403,
        341, 372, 373, 405, 343, 374, 375, 407, 345, 376, 377, 409, 347, 378, 379,
        411, 349, 380, 381, 413, 351, 382, 383, 415, 353, 384, 385, 417, 355, 386,
        387, 419, 357, 388, 389, 421, 359, 390, 391, 423, 361, 392, 393, 425, 363,
        394, 395, 427, 365, 396, 397, 429, 367, 398, 399, 431, 400, 432, 463, 464,
        402, 433, 434, 466, 404, 435, 436, 468, 406, 437, 438, 470, 408, 439, 440,
        472, 410, 441, 442, 474, 412, 443, 444, 476, 414, 445, 446, 478, 416, 447,
        448, 480, 418, 449, 450, 482, 420, 451, 452, 484, 422, 453, 454, 486, 424,
        455, 456, 488, 426, 457, 458, 490, 428, 459, 460, 492, 430, 461, 462, 494,
        465, 496, 497, 529, 467, 498, 499, 531, 469, 500, 501, 533, 471, 502, 503,
        535, 473, 504, 505, 537, 475, 506, 507, 539, 477, 508, 509, 541, 479, 510,
        511, 543, 481, 512, 513, 545, 483, 514, 515, 547, 485, 516, 517, 549, 487,
        518, 519, 551, 489, 520, 521, 553, 491, 522, 523, 555, 493, 524, 525, 557,
        495, 526, 527, 559, 528, 560, 591, 592, 530, 561, 562, 594, 532, 563, 564,
        596, 534, 565, 566, 598, 536, 567, 568, 600, 538, 569, 570, 602, 540, 571,
        572, 604, 542, 573, 574, 606, 544, 575, 576, 608, 546, 577, 578, 610, 548,
        579, 580, 612, 550, 581, 582, 614, 552, 583, 584, 616, 554, 585, 586, 618,
        556, 587, 588, 620, 558, 589, 590, 622, 593, 624, 625, 656, 595, 626, 627,
        658, 597, 628, 629, 660, 599, 630, 631, 662, 601, 632, 633, 663, 603, 634,
        635, 665, 605, 636, 637, 667, 607, 638, 639, 669, 609, 640, 641, 670, 611,
        642, 643, 672, 613, 644, 645, 674, 615, 646, 647, 676, 617, 648, 649, 677,
        619, 650, 651, 679, 621, 652, 653, 681, 623, 654, 655, 683, 657, 684, 685,
        708, 659, 686, 687, 710, 661, 688, 689, 712, 664, 690, 691, 713, 666, 692,
        693, 715, 668, 694, 695, 717, 671, 696, 697, 718, 673, 698, 699, 720, 675,
        700, 701, 722, 678, 702, 703, 723, 680, 704, 705, 725, 682, 706, 707, 727,
        709, 728, 729, 744, 711, 730, 731, 746, 714, 732, 733, 747, 716, 734, 735,
        749, 719, 736, 737, 750, 721, 738, 739, 752, 724, 740, 741, 753, 726, 742,
        743, 755, 745, 756, 757, 764, 748, 758, 759, 765, 751, 760, 761, 766, 754,
        762, 763, 767,
    ], dtype='int')
    
    assert np.all(ref41 == nnh.dgrade(4, 1))
    assert np.all(ref42 == nnh.dgrade(4, 2))
    assert np.all(ref81 == nnh.dgrade(8, 1))
    assert np.all(ref82 == nnh.dgrade(8, 2))
    assert np.all(ref84 == nnh.dgrade(8, 4))
