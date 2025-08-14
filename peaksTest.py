peaks = [['positive', 0.5248592675572212], ['negative', 0.71446485331341], ['positive', 0.9413919356484304], ['positive', 1.5267601270682047], ['negative', 1.6286746129082945], ['positive', 1.9706503377634519], ['negative', 2.906812969671189], ['positive', 3.2828068877264247], ['positive', 3.7711345182200726], ['negative', 3.9070482489363103], ['negative', 5.779162575082438], ['positive', 5.965504100227261], ['negative', 6.1949068861786065], ['positive', 6.293485181382413], ['negative', 6.444318598481123], 
['positive', 7.546922219396324], ['negative', 7.720747592814251], ['positive', 8.38076409474381], ['negative', 8.455355393911143], ['positive', 8.941075196298591], ['negative', 9.691133962966349], ['positive', 9.859949847399072], ['negative', 10.013760071185612]]
gestures = []
upToPositive = False
for peak in peaks:
    if peak[0] == 'negative':
        if upToPositive != False:
            gestures.append(f"Gesture from {upToPositive} to {peak[1]}")
            upToPositive = False
    if peak[0] == 'positive':
        if upToPositive == False:
            upToPositive = peak[1]
    else:
        upToPositive = False

#for g in range(len(gestures)):
#    gestures[g] = gestures[g]/16.466364913375998

print(gestures)