import ast

from common_imports import *
from scipy.stats import binom
from steadystate import getSteadyStateDist
import concurrent.futures
import csv


def findAllocationAB(nA, nB, diff):
    def calcReward(pA1, pA2, pA3, pB1, pB2, pB3):
        pASum = pA1 + pA2 + pA3
        pBSum = pB1 + pB2 + pB3

        PA1b = pA1 / pASum
        PA2b = pA2 / pASum
        PA3b = pA3 / pASum

        PB1l = pB1 / pBSum
        PB2l = pB2 / pBSum
        PB3l = pB3 / pBSum

        if diff == 0:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == 1:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (2 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (2 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (3 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (2 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (2 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (3 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (2 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (2 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (3 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == -1:
            ml = (2 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (2 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (3 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (2 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (2 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (3 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (2 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (2 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (3 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == 2:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (3 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (2 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (3 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (3 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (2 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (3 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (3 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (2 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (3 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == -2:
            ml = (3 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (2 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (3 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (3 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (2 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (3 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (3 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (2 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (3 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)
        if mr > ml:
            if mr > dm:
                # move right
                return 1, mr
            elif mr < dm:
                # don't move
                return 0, dm
            else:
                rand1 = random.randint(0, 1)
                if rand1 == 0:
                    return 1, mr
                else:
                    return 0, dm
        elif ml > mr:
            if ml > dm:
                # move left
                return -1, ml
            elif ml < dm:
                # don't move
                return 0, dm
            else:
                rand1 = random.randint(0, 1)
                if rand1 == 0:
                    return -1, ml
                else:
                    return 0, dm
        else:
            rand1 = random.randint(0, 1)
            if rand1 == 0:
                return -1, ml
            else:
                return 1, mr

    ps = [.33333, 0.5, .6]
    # A
    x = np.arange(0, nA + 1)
    PDFA1 = binom.pmf(x, nA, ps[0])
    PDFA2 = binom.pmf(x, nA, ps[1])
    PDFA3 = binom.pmf(x, nA, ps[2])

    # B
    x = np.arange(0, nB + 1)
    PDFB1 = binom.pmf(x, nB, ps[0])
    PDFB2 = binom.pmf(x, nB, ps[1])
    PDFB3 = binom.pmf(x, nB, ps[2])

    allocs = []
    rewards = []
    allChance = 0.0
    for i in range(len(PDFA1)):
        allocs.append([])
        rewards.append([])
        for j in range(len(PDFB1)):
            alloc, reward = calcReward(PDFA1[i], PDFA2[i], PDFA3[i], PDFB1[j], PDFB2[j], PDFB3[j])
            allocs[i].append(alloc)
            chance = ((1 / 3) * (PDFA1[i] + PDFA2[i] + PDFA3[i])) * ((1 / 3) * (PDFB1[j] + PDFB2[j] + PDFB3[j]))
            allChance += chance
            rewards[i].append(reward * chance)
    return allocs, rewards


def sumRewards(rewards):
    sum1 = 0.0
    for i in range(len(rewards)):
        sum1 += sum(rewards[i])
    return sum1


def findMax(n, diff, equiv):
    max = 0
    maxAlloc = [-1 - 1]
    rewardSums = []
    maxAllots = []
    if not equiv:
        if diff > 0:
            allocs, rewards = findAllocationAB(0, n, diff)
            rewSum = sumRewards(rewards)
            rewardSums.append(rewSum)
            max = rewSum
            maxAlloc = [0, n]
            maxAllots = allocs
            return max, maxAlloc, rewardSums, maxAllots
        elif diff < 0:
            allocs, rewards = findAllocationAB(n, 0, diff)
            rewSum = sumRewards(rewards)
            rewardSums.append(rewSum)
            max = rewSum
            maxAlloc = [n, 0]
            maxAllots = allocs
            return max, maxAlloc, rewardSums, maxAllots
        else:
            equalRec = math.floor(n / 2)
            allocs, rewards = findAllocationAB(equalRec, equalRec, diff)
            return sumRewards(rewards), [equalRec, equalRec], [sumRewards(rewards)], allocs
    else:
        equalRec = math.floor(n / 2)
        allocs, rewards = findAllocationAB(equalRec, equalRec, diff)
        return sumRewards(rewards), [equalRec, equalRec], [sumRewards(rewards)], allocs


def mapState(intState, environ, loc):
    return -(intState * 9) + environ * 3 + loc


def returnSurroundStates(intState):
    states = []
    for i in range(9):
        states.append(-(intState * 9) + i)
    return states


def stationaryDist(receptorMax, equiv, maxMI=False):
    def findsubPDF(A, B, maxAlloc, maxAllot):
        nA = maxAlloc[0]
        nB = maxAlloc[1]
        ps = [.33333, 0.5, .6]
        x = np.arange(0, nA + 1)
        if A == 1:
            GlobeAPDF = binom.pmf(x, nA, ps[0])  # conc. 1
        elif A == 2:
            GlobeAPDF = binom.pmf(x, nA, ps[1])  # conc. 2
        elif A == 3:
            GlobeAPDF = binom.pmf(x, nA, ps[2])  # conc. 3
        # B
        x = np.arange(0, nB + 1)
        if B == 1:
            GlobeBPDF = binom.pmf(x, nB, ps[0])  # conc. 1
        elif B == 2:
            GlobeBPDF = binom.pmf(x, nB, ps[1])  # conc. 2
        elif B == 3:
            GlobeBPDF = binom.pmf(x, nB, ps[2])  # conc. 3
        neg1, zero, one = 0.0, 0.0, 0.0
        for j in range(len(GlobeAPDF)):
            for k in range(len(GlobeBPDF)):
                if maxAllot[j][k] == -1:
                    neg1 += GlobeAPDF[j] * GlobeBPDF[k]
                elif maxAllot[j][k] == 0:
                    zero += GlobeAPDF[j] * GlobeBPDF[k]
                elif maxAllot[j][k] == 1:
                    one += GlobeAPDF[j] * GlobeBPDF[k]
        return [neg1 * (1 / 9), zero * (1 / 9), one * (1 / 9)]

    def findMI(maxAlloc, maxallots):
        A = [3, 2, 1, 3, 2, 1, 3, 2, 1]
        B = [1, 2, 3, 3, 1, 2, 2, 3, 1]
        pdfxy = []
        for i in range(len(A)):
            pdfxy.append(findsubPDF(A[i], B[i], maxAlloc, maxallots))
        sumMI = 0.0
        Px = 1 / 9
        Py = []
        for i in range(len(pdfxy[0])):
            Py.append(0.0)
        for i in range(len(pdfxy)):
            for j in range(len(pdfxy[0])):
                Py[j] += pdfxy[i][j]
        for i in range(len(pdfxy)):
            for j in range(len(pdfxy[0])):
                if Py[j] > 0:
                    sumMI += pdfxy[i][j] * math.log2(pdfxy[i][j] / (Px * Py[j]))
        return sumMI, pdfxy

    def findSyntI(maxAlloc):
        def findProbRow(A, B):
            nA = maxAlloc[0]
            nB = maxAlloc[1]
            ps = [.33333, 0.5, .6]
            x = np.arange(0, nA + 1)
            if A == 1:
                GlobeAPDF = binom.pmf(x, nA, ps[0])  # conc. 1
            elif A == 2:
                GlobeAPDF = binom.pmf(x, nA, ps[1])  # conc. 2
            elif A == 3:
                GlobeAPDF = binom.pmf(x, nA, ps[2])  # conc. 3
            # B
            x = np.arange(0, nB + 1)
            if B == 1:
                GlobeBPDF = binom.pmf(x, nB, ps[0])  # conc. 1
            elif B == 2:
                GlobeBPDF = binom.pmf(x, nB, ps[1])  # conc. 2
            elif B == 3:
                GlobeBPDF = binom.pmf(x, nB, ps[2])  # conc. 3
            return GlobeAPDF, GlobeBPDF

        def findIndMI(pxy):
            sumMI = 0.0
            Px = 1 / 9
            Py = []
            for i in range(len(pxy[0])):
                Py.append(0.0)
            for i in range(len(pxy)):
                for j in range(len(pxy[0])):
                    Py[j] += pxy[i][j]
            for i in range(len(pxy)):
                for j in range(len(pxy[0])):
                    if Py[j] > 0 and pxy[i][j] > 0:
                        sumMI += pxy[i][j] * math.log2(pxy[i][j] / (Px * Py[j]))
            return sumMI

        As = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        Bs = [3, 2, 1, 3, 2, 1, 3, 2, 1]
        pxyA = []
        pxyB = []
        for i in range(len(As)):
            PDFA, PDFB = findProbRow(As[i], Bs[i])
            pxyA.append(PDFA)
            pxyB.append(PDFB)

        return findIndMI(pxyA) + findIndMI(pxyB)

    A = [3, 2, 1, 3, 2, 1, 3, 2, 1]
    B = [1, 2, 3, 3, 1, 2, 2, 3, 1]
    # receptorMax = 200
    largeIntState = 5
    N = largeIntState * 9
    transProbs = np.zeros((2 * N + 3 * 9, 2 * N + 3 * 9))
    curLoc = 0  # location - 1
    curloc = 2
    max, maxAlloc, rewardSums, maxallots = findMax(receptorMax, curloc, equiv)
    print("done here")
    newStateBase = [[4, 2, 0], [2, 3, 1], [3, 1, 2]]
    A = [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
    B = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    for i in range(largeIntState):
        for j in range(3):
            newIntStates = [x + i for x in newStateBase[j]]
            for k in range(3):
                curloc = (i * largeIntState) + j * 3 + k
                # print(curloc)
                currA = A[j][k]
                currB = B[j][k]
                loc = k
                nA = maxAlloc[0]
                nB = maxAlloc[1]
                ps = [.33333, 0.5, .6]
                x = np.arange(0, nA + 1)
                # A
                PDFA1 = binom.pmf(x, nA, ps[0])  # conc. 1
                PDFA2 = binom.pmf(x, nA, ps[1])  # conc. 2
                PDFA3 = binom.pmf(x, nA, ps[2])  # conc. 3
                hash_State = {}
                # B
                x = np.arange(0, nB + 1)
                PDFB1 = binom.pmf(x, nB, ps[0])  # conc. 1
                PDFB2 = binom.pmf(x, nB, ps[1])  # conc. 2
                PDFB3 = binom.pmf(x, nB, ps[2])  # conc. 3
                if currA == 3:
                    PDFA = PDFA3
                elif currA == 2:
                    PDFA = PDFA2
                elif currA == 1:
                    PDFA = PDFA1
                if currB == 3:
                    PDFB = PDFB3
                elif currB == 2:
                    PDFB = PDFB2
                elif currB == 1:
                    PDFB = PDFB1

                for l in range(len(PDFA)):
                    for m in range(len(PDFB)):
                        curProb = PDFA[l] * PDFB[m]
                        curNewState = newIntStates[(k + maxallots[l][m]) % 3]
                        if hash_State.get(curNewState) is None:
                            hash_State[curNewState] = 0
                        hash_State[curNewState] += curProb
                for key in hash_State:
                    # print()
                    # print(i)
                    loc = mapState(i + 2, j, k)
                    # print(loc)
                    loc = N + loc + 9
                    # print("2 cur loc: " + str(loc))
                    # if loc == 0:
                    #    print(str(i) + ", " + str(j) +", " + str(k))
                    states = returnSurroundStates(key)
                    for state in states:
                        state = N + state + 9
                        if state >= len(transProbs[0]):
                            state = len(transProbs[0]) - 1
                        if state < 0:
                            state = 0
                        if loc < 0 or loc >= len(transProbs[0]):
                            print("loc 2: " + str(loc))
                        if state < 0 or state >= len(transProbs[0]):
                            print("state 2: " + str(state))
                        transProbs[loc][state] += hash_State[key] / 9
    max2 = max
    # finding MI given state >2
    MI2, pdfxy2 = findMI(maxAlloc, maxallots)
    MIsynt2 = findSyntI(maxAlloc)

    # -------------------------
    curloc = -2
    max, maxAlloc, rewardSums, maxallots = findMax(receptorMax, curloc, equiv)
    newStateBase = [[0, -2, -4], [-2, -1, -3], [-1, -3, -2]]
    A = [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
    B = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    for i in range(largeIntState):
        for j in range(3):
            newIntStates = [x - i for x in newStateBase[j]]
            for k in range(3):
                # curloc = (i*largeIntState)+j*3+k
                # print(curloc)
                currA = A[j][k]
                currB = B[j][k]
                loc = k
                nA = maxAlloc[0]
                nB = maxAlloc[1]
                ps = [.33333, 0.5, .6]
                x = np.arange(0, nA + 1)
                # A
                PDFA1 = binom.pmf(x, nA, ps[0])  # conc. 1
                PDFA2 = binom.pmf(x, nA, ps[1])  # conc. 2
                PDFA3 = binom.pmf(x, nA, ps[2])  # conc. 3
                hash_State = {}
                # B
                x = np.arange(0, nB + 1)
                PDFB1 = binom.pmf(x, nB, ps[0])  # conc. 1
                PDFB2 = binom.pmf(x, nB, ps[1])  # conc. 2
                PDFB3 = binom.pmf(x, nB, ps[2])  # conc. 3
                if currA == 3:
                    PDFA = PDFA3
                elif currA == 2:
                    PDFA = PDFA2
                elif currA == 1:
                    PDFA = PDFA1
                if currB == 3:
                    PDFB = PDFB3
                elif currB == 2:
                    PDFB = PDFB2
                elif currB == 1:
                    PDFB = PDFB1

                for l in range(len(PDFA)):
                    for m in range(len(PDFB)):
                        curProb = PDFA[l] * PDFB[m]
                        curNewState = newIntStates[(k + maxallots[l][m]) % 3]
                        if hash_State.get(curNewState) is None:
                            hash_State[curNewState] = 0
                        hash_State[curNewState] += curProb
                for key in hash_State:
                    loc = mapState(-(i + 4), j, k)
                    loc = N + loc - 9
                    # print("-2 cur loc: " + str(loc))
                    states = returnSurroundStates(key)
                    for state in states:
                        state = N + state + 9
                        # print(newIntStates)
                        # print(loc)
                        # print(state)
                        # print()
                        if state >= len(transProbs[0]):
                            # print(state)
                            state = len(transProbs[0]) - 1
                        if state < 0:
                            # print(state)
                            state = 0
                        if loc < 0 or loc >= len(transProbs[0]):
                            print("loc -2: " + str(loc))
                        if state < 0 or state >= len(transProbs[0]):
                            print("state -2: " + str(state))
                        transProbs[loc][state] += hash_State[key] / 9
    maxneg2 = max
    MIneg2, pdfxyneg2 = findMI(maxAlloc, maxallots)
    MIsyntneg2 = findSyntI(maxAlloc)

    curloc = 1
    max, maxAlloc, rewardSums, maxallots = findMax(receptorMax, curloc, equiv)
    newStateBase = [[3, 1, -1], [1, 2, 0], [2, 0, 1]]
    A = [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
    B = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    for j in range(3):
        newIntStates = newStateBase[j]
        for k in range(3):
            # curloc = (i*largeIntState)+j*3+k
            # print(curloc)
            currA = A[j][k]
            currB = B[j][k]
            loc = k
            nA = maxAlloc[0]
            nB = maxAlloc[1]
            ps = [.33333, 0.5, .6]
            x = np.arange(0, nA + 1)
            # A
            PDFA1 = binom.pmf(x, nA, ps[0])  # conc. 1
            PDFA2 = binom.pmf(x, nA, ps[1])  # conc. 2
            PDFA3 = binom.pmf(x, nA, ps[2])  # conc. 3
            hash_State = {}
            # B
            x = np.arange(0, nB + 1)
            PDFB1 = binom.pmf(x, nB, ps[0])  # conc. 1
            PDFB2 = binom.pmf(x, nB, ps[1])  # conc. 2
            PDFB3 = binom.pmf(x, nB, ps[2])  # conc. 3
            if currA == 3:
                PDFA = PDFA3
            elif currA == 2:
                PDFA = PDFA2
            elif currA == 1:
                PDFA = PDFA1
            if currB == 3:
                PDFB = PDFB3
            elif currB == 2:
                PDFB = PDFB2
            elif currB == 1:
                PDFB = PDFB1

            for l in range(len(PDFA)):
                for m in range(len(PDFB)):
                    curProb = PDFA[l] * PDFB[m]
                    curNewState = newIntStates[(k + maxallots[l][m]) % 3]
                    if hash_State.get(curNewState) is None:
                        hash_State[curNewState] = 0
                    hash_State[curNewState] += curProb
            for key in hash_State:
                loc = mapState(1, j, k)
                loc = N + loc + 9
                # print("1 cur loc: " + str(loc))
                states = returnSurroundStates(key)
                for state in states:
                    state = N + state + 9
                    if state >= len(transProbs[0]):
                        state = len(transProbs[0]) - 1
                    if state < 0:
                        state = 0
                    if loc < 0 or loc >= len(transProbs[0]):
                        print("loc 1: " + str(loc))
                    if state < 0 or state >= len(transProbs[0]):
                        print("state 1: " + str(state))
                    transProbs[loc][state] += hash_State[key] / 9
    max1 = max
    MI1, pdfxy1 = findMI(maxAlloc, maxallots)
    MIsynt1 = findSyntI(maxAlloc)

    curloc = -1
    max, maxAlloc, rewardSums, maxallots = findMax(receptorMax, curloc, equiv)
    newStateBase = [[1, -1, -3], [-1, 0, -2], [0, -2, -1]]
    A = [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
    B = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    for j in range(3):
        newIntStates = newStateBase[j]
        for k in range(3):
            # curloc = (i*largeIntState)+j*3+k
            # print(curloc)
            currA = A[j][k]
            currB = B[j][k]
            loc = k
            nA = maxAlloc[0]
            nB = maxAlloc[1]
            ps = [.33333, 0.5, .6]
            x = np.arange(0, nA + 1)
            # A
            PDFA1 = binom.pmf(x, nA, ps[0])  # conc. 1
            PDFA2 = binom.pmf(x, nA, ps[1])  # conc. 2
            PDFA3 = binom.pmf(x, nA, ps[2])  # conc. 3
            hash_State = {}
            # B
            x = np.arange(0, nB + 1)
            PDFB1 = binom.pmf(x, nB, ps[0])  # conc. 1
            PDFB2 = binom.pmf(x, nB, ps[1])  # conc. 2
            PDFB3 = binom.pmf(x, nB, ps[2])  # conc. 3
            if currA == 3:
                PDFA = PDFA3
            elif currA == 2:
                PDFA = PDFA2
            elif currA == 1:
                PDFA = PDFA1
            if currB == 3:
                PDFB = PDFB3
            elif currB == 2:
                PDFB = PDFB2
            elif currB == 1:
                PDFB = PDFB1

            for l in range(len(PDFA)):
                for m in range(len(PDFB)):
                    curProb = PDFA[l] * PDFB[m]
                    curNewState = newIntStates[(k + maxallots[l][m]) % 3]
                    if hash_State.get(curNewState) is None:
                        hash_State[curNewState] = 0
                    hash_State[curNewState] += curProb
            for key in hash_State:
                loc = mapState(-3, j, k)
                loc = N + loc - 9
                # print("-1 cur loc: " + str(loc))
                states = returnSurroundStates(key)
                for state in states:
                    state = N + state + 9
                    if state >= len(transProbs[0]):
                        state = len(transProbs[0]) - 1
                    if state < 0:
                        state = 0

                    if loc < 0 or loc >= len(transProbs[0]):
                        print("loc -1: " + str(loc))
                    if state < 0 or state >= len(transProbs[0]):
                        print("state -1: " + str(state))
                    transProbs[loc][state] += hash_State[key] / 9
    maxneg1 = max
    MIneg1, pdfxyneg1 = findMI(maxAlloc, maxallots)
    MIsyntneg1 = findSyntI(maxAlloc)

    curloc = 0
    max, maxAlloc, rewardSums, maxallots = findMax(receptorMax, curloc, equiv)
    newStateBase = [[2, 0, -2], [0, 1, -1], [1, -1, 0]]
    A = [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
    B = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    for j in range(3):
        newIntStates = newStateBase[j]
        for k in range(3):
            # curloc = (i*largeIntState)+j*3+k
            # print(curloc)
            currA = A[j][k]
            currB = B[j][k]
            loc = k
            nA = maxAlloc[0]
            nB = maxAlloc[1]
            ps = [.33333, 0.5, .6]
            x = np.arange(0, nA + 1)
            # A
            PDFA1 = binom.pmf(x, nA, ps[0])  # conc. 1
            PDFA2 = binom.pmf(x, nA, ps[1])  # conc. 2
            PDFA3 = binom.pmf(x, nA, ps[2])  # conc. 3
            hash_State = {}
            # B
            x = np.arange(0, nB + 1)
            PDFB1 = binom.pmf(x, nB, ps[0])  # conc. 1
            PDFB2 = binom.pmf(x, nB, ps[1])  # conc. 2
            PDFB3 = binom.pmf(x, nB, ps[2])  # conc. 3
            if currA == 3:
                PDFA = PDFA3
            elif currA == 2:
                PDFA = PDFA2
            elif currA == 1:
                PDFA = PDFA1
            if currB == 3:
                PDFB = PDFB3
            elif currB == 2:
                PDFB = PDFB2
            elif currB == 1:
                PDFB = PDFB1

            for l in range(len(PDFA)):
                for m in range(len(PDFB)):
                    curProb = PDFA[l] * PDFB[m]
                    curNewState = newIntStates[(k + maxallots[l][m]) % 3]
                    if hash_State.get(curNewState) is None:
                        hash_State[curNewState] = 0
                    hash_State[curNewState] += curProb
            for key in hash_State:
                loc = mapState(0, j, k)
                loc = N + loc + 9
                # print("0 cur loc: " + str(loc))
                states = returnSurroundStates(key)
                for state in states:
                    state = N + state + 9
                    if state >= len(transProbs[0]):
                        state = len(transProbs[0]) - 1
                    if state < 0:
                        state = 0
                    if loc < 0 or loc >= len(transProbs[0]):
                        print("loc 0: " + str(loc))
                    if state < 0 or state >= len(transProbs[0]):
                        print("state 0: " + str(state))
                    transProbs[loc][state] += hash_State[key] / 9
    max0 = max
    MI0, pdfxy0 = findMI(maxAlloc, maxallots)
    MIsynt0 = findSyntI(maxAlloc)

    # for arr in transProbs:
    #    print(arr)
    # exit()
    import scipy.io

    # Save to MAT file
    scipy.io.savemat("transProbs.mat", {'array': transProbs})

    for i in range(len(transProbs)):
        ret = ""
        curSum = sum(transProbs[i])
        if round(curSum * 10000) / 10000 != 1:
            print(str(sum(transProbs[i])) + "looks weird at: " + str(i))
        for j in range(len(transProbs[i])):
            ret += str(transProbs[i][j]) + ", "
    # exit()
    # print(ret)
    front = [1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0]
    middle = [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0,
              3.0, 2.0, 1.0, 3.0, 2.0, 1.0]
    back = [3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0]
    # largeIntState = 5
    # N = largeIntState * 9
    # 2*N + 3*9
    rewards = []
    for i in range(largeIntState):
        rewards = rewards + front
    rewards = rewards + middle

    for i in range(largeIntState):
        rewards = rewards + back

    front = [max2]
    middle = [max1, max0, maxneg1]
    back = [maxneg2]
    rewardsQuant = []
    for i in range(largeIntState):
        rewardsQuant = rewardsQuant + front
    rewardsQuant = rewardsQuant + middle
    for i in range(largeIntState):
        rewardsQuant = rewardsQuant + back

    front = [MI2]
    middle = [MI1, MI0, MIneg1]
    back = [MIneg2]
    MIs = []
    for i in range(largeIntState):
        MIs = MIs + front
    MIs = MIs + middle
    for i in range(largeIntState):
        MIs = MIs + back

    front = [pdfxy2]
    middle = [pdfxy1, pdfxy0, pdfxyneg1]
    back = [pdfxyneg2]
    pdfs = []
    for i in range(largeIntState):
        pdfs = pdfs + front
    pdfs = pdfs + middle
    for i in range(largeIntState):
        pdfs = pdfs + back

    front = [MIsynt2]
    middle = [MIsynt1, MIsynt0, MIsyntneg1]
    back = [MIsyntneg2]
    pdfsSynt = []
    for i in range(largeIntState):
        pdfsSynt = pdfsSynt + front
    pdfsSynt = pdfsSynt + middle
    for i in range(largeIntState):
        pdfsSynt = pdfsSynt + back

    # a row-stochastic three-state Markov transition probability matrix
    # P = np.array([[0.9,0.07,0.03],[0.1,0.8,0.1],[0.15,0.25,0.6]])

    q = getSteadyStateDist(transProbs)

    # plt.plot(q)
    # plt.show()
    # exit()

    front = [3]
    middle = [(2 + 3 + 3) / 3, (2 + 2 + 3) / 3, (2 + 3 + 3) / 3]
    back = [3]
    best = []
    for i in range(largeIntState):
        best = best + front
    best = best + middle
    for i in range(largeIntState):
        best = best + back

    bestGrowth = 0.0

    def sum_of_subarrays(array, chunk_size=9):
        return [sum(array[i:i + chunk_size]) for i in range(0, len(array), chunk_size)]

    quantized = sum_of_subarrays(q)

    pdfsfinal = []
    for j in range(len(pdfs[0])):
        pdfsfinal.append([])
        for k in range(len(pdfs[0][j])):
            pdfsfinal[j].append(0.0)

    for i in range(len(quantized)):
        for j in range(len(pdfs[i])):
            for k in range(len(pdfs[i][j])):
                pdfsfinal[j][k] += pdfs[i][j][k] * quantized[i]

    sum2 = 0.0
    for i in range(len(pdfsfinal)):
        for j in range(len(pdfsfinal[i])):
            sum2 += pdfsfinal[i][j]

    sumMISyn = 0.0
    Px = 1 / 9
    Py = []
    for i in range(len(pdfsfinal[0])):
        Py.append(0.0)
    for i in range(len(pdfsfinal)):
        for j in range(len(pdfsfinal[0])):
            Py[j] += pdfsfinal[i][j]
    for i in range(len(pdfsfinal)):
        for j in range(len(pdfsfinal[0])):
            if Py[j] > 0 and pdfsfinal[i][j] > 0:
                sumMISyn += pdfsfinal[i][j] * math.log2(pdfsfinal[i][j] / (Px * Py[j]))
    # return sumMISyn, pdfsfinal

    sum1 = 0.0
    print(len(quantized))
    print(len(rewardsQuant))
    for i in range(len(quantized)):
        sum1 += quantized[i] * rewardsQuant[i]
        bestGrowth += quantized[i] * best[i]

    totMI = 0.0
    syntacticMI = 0.0
    for i in range(len(quantized)):
        totMI += quantized[i] * MIs[i]
        syntacticMI += pdfsSynt[i] * quantized[i]
    print("Reward: " + str(sum1))
    print("MI: " + str(totMI))
    return sum1, totMI, sumMISyn, syntacticMI


recList = []
currSum = 2
indexAdd = 0
appendAdd = 2
for i in range(50):
    recList.append(currSum)
    if i % 2 == 0:
        indexAdd += appendAdd
    currSum += indexAdd
#run below to generate new data
"""
rewardsAdapt = []
MIAdapt = []
rewardsEquiv = []
MIEquiv = []
MIsynAdapt = []
MIsynEquiv = []
syntacticTotAdapt = []
syntacticTotEquiv = []
for i in range(len(recList)):
    print(recList[i])
    curReward, curMIAdapt, curMIsynAdapt, cursyntacticMIAdapt = stationaryDist(recList[i], False)
    curRewardequiv, curMIEquiv, curMIsynEquiv, cursyntacticMIEquiv = stationaryDist(recList[i], True)
    rewardsAdapt.append(curReward)
    #rewardsAdapt.append(curReward)
    rewardsEquiv.append(curRewardequiv)
    #rewardsEquiv.append(curRewardequiv)
    MIEquiv.append(curMIEquiv)
    MIAdapt.append(curMIAdapt)
    MIsynAdapt.append(curMIsynAdapt)
    MIsynEquiv.append(curMIsynEquiv)
    syntacticTotAdapt.append(cursyntacticMIAdapt)
    syntacticTotEquiv.append(cursyntacticMIEquiv)

print("rewardsAdapt = " + str(rewardsAdapt))
print("rewardsEquiv = " + str(rewardsEquiv))
print("MIAdapt = " + str(MIAdapt))
print("MIEquiv = " + str(MIEquiv))
print("MIsynAdapt = " + str(MIsynAdapt))
print("MIsynEquiv = " + str(MIsynEquiv))
print("syntacticTotAdapt = " + str(syntacticTotAdapt))
print("syntacticTotEquiv = " + str(syntacticTotEquiv))

exit()
"""
#pre generated data
# region
rewardsAdapt = [2.054565606915843, 2.0952492211224953, 2.125833488892759, 2.1689667917479563, 2.2006176117338865,
                2.2375540593792294, 2.264697635732326, 2.287091813505161, 2.315430889797989, 2.3355562755204406,
                2.351806002043567, 2.369577392565214, 2.389407517950968, 2.4035499324807454, 2.4134691777727184,
                2.4233505943002545, 2.4321470028953383, 2.4386262566192354, 2.445819867369395, 2.4551087902962037,
                2.4586885555663835, 2.463816271319982, 2.467685560920451, 2.472654670858992, 2.475912984581654,
                2.4780831017730924, 2.4805324636598436, 2.483496423418372, 2.4836325642591985, 2.4865502571846125,
                2.4874377732039954, 2.489143705454896, 2.4890451732927805, 2.4909074471944352, 2.4913753500114084,
                2.4922842110995944, 2.493123428418222, 2.493379335798589, 2.4943279325433894, 2.4948099290410526,
                2.4952182061180133, 2.4953318464011347, 2.4958839386570766, 2.496147698197042, 2.4963812244404,
                2.496725566472161, 2.4970072209003216, 2.4971421707533645, 2.497265725447539, 2.497450646981115]
rewardsEquiv = [2.042315498105036, 2.0643911349510087, 2.0922183270467882, 2.1271875532718663, 2.1552975008248434,
                2.1924750135491755, 2.218600415780351, 2.24563291996613, 2.271724261784934, 2.296297096078148,
                2.315802231334029, 2.3368107434998144, 2.3578551036354147, 2.37532044412631, 2.3873420223127133,
                2.3977777558522924, 2.408837829169409, 2.4182464445322194, 2.427645205055182, 2.438201863760847,
                2.4432387910382363, 2.450101964120588, 2.4548550537323788, 2.459700557458231, 2.4651277018014515,
                2.468297641842974, 2.471189488442913, 2.475358332814887, 2.476503740784133, 2.4797341156725223,
                2.4810168191042874, 2.483435834237246, 2.4839077223743096, 2.4861216956617094, 2.486409717556011,
                2.4878238991952837, 2.4891334851414064, 2.490209541573774, 2.491380021208179, 2.4921619640970345,
                2.4928392769614063, 2.4932673105823997, 2.493790621715351, 2.494280452243838, 2.494944008642472,
                2.495414580095674, 2.4958725765294703, 2.4961381584902402, 2.496288617111255, 2.496684710260775]
MIAdapt = [0.05006980720803111, 0.08303886455898919, 0.13333329392438714, 0.20285540295402274, 0.26486501218526703,
           0.3485448639942033, 0.41803583015390117, 0.48073900172207473, 0.5497113927903298, 0.6075878079132333,
           0.6540787549951836, 0.7108176508620835, 0.7774642266802035, 0.8302839815647971, 0.8723815751325711,
           0.920783340981565, 0.9712864334273352, 1.0065178942965147, 1.047284332663921, 1.0896226409882366,
           1.1230463426431005, 1.1600849903412958, 1.1868848518453907, 1.215371421978937, 1.2343379237546919,
           1.2556536240147032, 1.282868992872144, 1.2994369396465661, 1.316259779593651, 1.330439033482663,
           1.3469590262219282, 1.3590480207212283, 1.3728959667490908, 1.381668980374553, 1.3931124624628228,
           1.403296729515281, 1.4125821241218146, 1.4217314609973508, 1.4287752909161011, 1.4346952383365976,
           1.4395453749161051, 1.446027292077698, 1.4503304448607424, 1.4550850090655405, 1.4588693964778716,
           1.4627955556613235, 1.4656005606800036, 1.468961186140625, 1.4720320833636966, 1.4744364221109658]
MIEquiv = [0.042246446351629434, 0.04783191974656454, 0.0866722617054747, 0.12855347664715916, 0.1719322572912571,
           0.22088876989712788, 0.2830542981684806, 0.33397302271539636, 0.3925414287108168, 0.45223485356465537,
           0.4988902348673505, 0.5538994576843309, 0.6122260005808776, 0.6622070594374475, 0.7025126340641523,
           0.7433731622956924, 0.7895913543333329, 0.8321313466259841, 0.8677955478671533, 0.9106709382764355,
           0.9499126410176829, 0.9834548225545634, 1.0135077270854447, 1.0461975320744636, 1.070370226734837,
           1.0967000384202141, 1.1250987420767495, 1.147867276151903, 1.1719704403922047, 1.1874335175129835,
           1.207252471537193, 1.225271400474772, 1.2429464118322158, 1.2564792267728253, 1.2716849299118407,
           1.2854169420219064, 1.2982707975570957, 1.311239515915048, 1.3211665585117067, 1.330285473154444,
           1.3383469914801809, 1.3484908400497233, 1.355684650782976, 1.3632957061995596, 1.369616356693553,
           1.3753187077860776, 1.3808540547838428, 1.3863356311597677, 1.3914376358284695, 1.3955735960631444]
MIsynAdapt = [0.02731468337927127, 0.06069525624442264, 0.06361750850978232, 0.10777801731576857, 0.1405361713092033,
              0.1889511465443326, 0.2154100364702345, 0.24972520650689908, 0.29271352141126394, 0.3224945793151505,
              0.34773797257775424, 0.3738488174038814, 0.41370919407651696, 0.4453600932184375, 0.46173629369817515,
              0.4830368859193179, 0.5028246733105208, 0.5220550511252923, 0.5410055820268442, 0.5690739052271847,
              0.5803279976896099, 0.5954114878792083, 0.6089852118409752, 0.627467675665084, 0.6394426221865871,
              0.6493868408648737, 0.6606037564506603, 0.6733183582461925, 0.6773787347420921, 0.6887309727878054,
              0.6947271227738878, 0.7035831844148388, 0.7062341350399448, 0.7148503631999327, 0.719132370989352,
              0.7248027233306675, 0.7302110783528404, 0.7332613534556898, 0.7385991962766142, 0.7419774376443697,
              0.7448413186552068, 0.746915857650454, 0.750336279658096, 0.7525572882727297, 0.7545959880847658,
              0.7569665279786514, 0.7589684858600406, 0.7603691716808583, 0.7616350458468457, 0.7631164140387244]
MIsynEquiv = [0.017830936682223367, 0.03584594556598736, 0.037917195079668925, 0.06231741103350705, 0.08750028723829616,
              0.13024317020663229, 0.15202337445171782, 0.19057931674587275, 0.2240943280324426, 0.2595441384343601,
              0.2929279022509297, 0.3315652680850105, 0.36661223857249825, 0.4042247861766387, 0.4250054396192636,
              0.4444373414334225, 0.4708037344396764, 0.49467799219918845, 0.5174749424908985, 0.5489811071821464,
              0.5638580690735752, 0.5845882067756298, 0.5997944267960993, 0.6165626466566981, 0.6355264453427886,
              0.6487767838940872, 0.6624247651663318, 0.6778210651675867, 0.6864318581801369, 0.697790452854865,
              0.7061317444104693, 0.7166563197489534, 0.7222680816147107, 0.73167423572782, 0.7350219096959045,
              0.7426661731440815, 0.7496665550317083, 0.756892844195842, 0.7630549625678135, 0.7674104514252901,
              0.7711507954168603, 0.7747845076799773, 0.7778164005870757, 0.7811696460840241, 0.7851080825001364,
              0.7880285382767841, 0.7909317170238988, 0.7929956833710371, 0.794349772678167, 0.7969752587603408]
syntacticTotAdapt = [0.6256216582927425, 1.1998366017729274, 1.730965213126063, 2.6892736167079483, 3.5368201344704553,
                     4.640019697334184, 5.597580610675518, 6.699317231710602, 7.639095487095315, 8.667475230164206,
                     9.559045279634224, 10.471871569118566, 11.275975168395776, 12.102676724889086, 12.816775754362853,
                     13.533658422971321, 14.174665797359348, 14.815562685703034, 15.361450341973878, 15.900705620113257,
                     16.389705536278026, 16.83727345778661, 17.23935887815646, 17.62999208173402, 17.958204663316145,
                     18.29229060323852, 18.593786613245182, 18.860551218708032, 19.120689776855034, 19.324652516852673,
                     19.53136768639536, 19.715984678397778, 19.89422685816022, 20.040434842113903, 20.180545172179603,
                     20.308192244361077, 20.418984559058604, 20.52945080375206, 20.618102601730236, 20.699136880550743,
                     20.7697974397143, 20.841593125069764, 20.898553332200084, 20.9538747853638, 21.00042138538397,
                     21.043060909809444, 21.079294804845272, 21.114770380361737, 21.145407270411987, 21.172041435895547]
syntacticTotEquiv = [0.6388825003415821, 1.2472244324600923, 1.8276098507133272, 2.913099983824612, 3.9099740346911864,
                     5.263753729126865, 6.474042554688983, 7.904045529833444, 9.161888840974507, 10.539530093228084,
                     11.743279661424255, 13.005626056989856, 14.109530597762626, 15.23858419893879, 16.23053671957343,
                     17.229343967605715, 18.110949993213147, 18.98793220084437, 19.76390868320868, 20.527209309793577,
                     21.202775913288548, 21.860361391522282, 22.441888010015315, 23.002721601768346, 23.498257158297267,
                     23.97251747374923, 24.391385953909094, 24.78983383785218, 25.141771968737157, 25.474925126959,
                     25.769294996258434, 26.04681795298932, 26.292115875986745, 26.522519682683036, 26.72617669448485,
                     26.916766472648955, 27.085151262930157, 27.24212219296989, 27.38065210451343, 27.509249025128423,
                     27.622538338364308, 27.727222675178048, 27.819225412473966, 27.903819825354592, 27.977947328076738,
                     28.04574797304972, 28.10495696856849, 28.158815008863137, 28.205670310542857, 28.24804970235793]
# endregion
curReward, curMIAdapt, curMIsynAdapt, cursyntacticMI = stationaryDist(100, False)
curRewardequiv, curMIEquiv, curMIsynEquiv, cursyntacticMI = stationaryDist(100, True)

c1 = 2
c2 = 0.001
for i in range(len(rewardsAdapt)):
    rewardsAdapt[i] = rewardsAdapt[i] - c1 - recList[i] * c2
    rewardsEquiv[i] = rewardsEquiv[i] - c1 - recList[i] * c2

oldreclist = recList
recList = [math.log10(n) for n in recList]
plt.rc('pdf', fonttype=42)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
scatter = plt.scatter(MIAdapt, rewardsAdapt, label="$\hat{R}_{Adaptive}$", c=recList, cmap='viridis', marker='s',
                      edgecolor='black', alpha=0.5, zorder=2)
plt.scatter(MIEquiv, rewardsEquiv, label="$\hat{R}_{Equivalent}$", c=recList, cmap='viridis', alpha=0.5, zorder=2)
plt.scatter(curMIAdapt, curReward - c1 - 100 * c2, label="$\hat{R}_{Adaptive}$ ($N_{tot} = 100$)", color='orange',
            marker='s', edgecolor='black', s=100, zorder=2)
plt.scatter(curMIEquiv, curRewardequiv - c1 - 100 * c2, label="$\hat{R}_{Equivalent}$ ($N_{tot} = 100$)", color='blue',
            s=100, zorder=2)
cbar = plt.colorbar(scatter, label='Receptors (Log Scale, Base 10)')
cbar.set_label('Receptors $N_{tot}$ (Log Scale, Base 10)', fontsize=12)
value_ranges = ["0", "600", "1300"]  # Adjust these based on your data
colors = [scatter.cmap(scatter.norm(value)) for value in (0, 600, 1300)]  # Replace with actual values or percentiles
patches = [mpatches.Patch(color=colors[i], label=value_ranges[i]) for i in range(len(value_ranges))]
plt.legend(handles=patches, fontsize=16)
plt.ylabel("$\hat{R}_{Strategy}$", fontsize=15)
plt.xlabel("$I_U|_{strategy}$ [bits]", fontsize=15)
plt.grid(zorder=1)
plt.legend(fontsize=11)
plt.show()

recList = oldreclist
rewardDiff = [a - b for a, b in zip(rewardsAdapt, rewardsEquiv)]
MIDiff = [a - b for a, b in zip(MIAdapt, MIEquiv)]
rewardPerReceptor = []
for i in range(len(rewardDiff)):
    rewardPerReceptor.append(rewardDiff[i] / MIDiff[i])
plt.clf()
scatter = plt.scatter(recList, rewardPerReceptor, label="SI", zorder=2)
plt.ylabel(r"$\frac{\hat{R}_{Adaptive} - \hat{R}_{Equivalent}}{I_{subj}}$" "\n" "[reward/bit]", rotation=0, fontsize=17,
           labelpad=55)
plt.gca().yaxis.set_label_coords(-0.24, 0.2)
plt.xlabel("Receptor count $N_{tot}$ (Log Scale, Base 10)", fontsize=14)
plt.xscale('log')
ticks = [10, 100, 1000]
plt.gca().set_xticks(ticks)
plt.gca().set_xticklabels(['10', '100', '1000'])
plt.grid(zorder=1)
plt.legend(bbox_to_anchor=(5, 1))
plt.tight_layout()
plt.show()
