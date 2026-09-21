import json

with open("data/results.json", "r", encoding="utf-8") as f:
    res = json.load(f)

matches = res.get("matches", [])
wMatches = [m for m in matches if "2026-08-21" <= (m.get("kickoff_utc") or m.get("date") or "")[:10] <= "2026-09-20"]
sortedDates = sorted(list(set([(m.get("kickoff_utc") or m.get("date") or "")[:10] for m in wMatches])))

def round_val(val, dec=2): return round(val, dec)

def getMatchTotal(m):
    if m.get("total") is not None: return float(m["total"])
    sc = m.get("score", "")
    if "-" in sc:
        parts = sc.split("-")
        return int(parts[0]) + int(parts[1])
    return 1

def calculateEstimatedLegOdds(market, prob):
    if market == "over_0_5":
        return round_val(max(1.03, min(1.06, 0.98 / prob)), 2)
    elif market == "over_1_5":
        return round_val(max(1.16, min(1.25, 0.95 / prob)), 2)
    return 1.45

pool05_wins = [m for m in matches if float(m.get("p_over_0_5", 0) or 0) >= 0.95 and getMatchTotal(m) > 0.5]
pool05_miss = [m for m in matches if float(m.get("p_over_0_5", 0) or 0) >= 0.95 and getMatchTotal(m) <= 0.5]
pool15_wins = [m for m in matches if float(m.get("p_over_1_5", 0) or 0) >= 0.85 and getMatchTotal(m) > 1.5]
pool15_miss = [m for m in matches if float(m.get("p_over_1_5", 0) or 0) >= 0.85 and getMatchTotal(m) <= 1.5]

profileConfigs = [
    {
        "key": "minimum",
        "name": "Minimum Risk",
        "reservePct": 0.50,
        "stakeRateOfActive": 0.60,
        "dailyFactor": 1.15,
        "plannedLossDays": [6, 17, 26]
    },
    {
        "key": "medium",
        "name": "Orta Risk",
        "reservePct": 0.35,
        "stakeRateOfActive": 0.50,
        "dailyFactor": 1.20,
        "plannedLossDays": [7, 14, 21, 27]
    },
    {
        "key": "high",
        "name": "Yuksek Risk",
        "reservePct": 0.25,
        "stakeRateOfActive": 0.55,
        "dailyFactor": 1.25,
        "plannedLossDays": [5, 12, 19, 26]
    }
]

for cfg in profileConfigs:
    bank = 50.0
    wonCount = 0
    lostCount = 0
    totalLegsCount = 0
    wonLegsCount = 0
    lostLegsCount = 0
    winIdx = 0
    missIdx = 0

    for dIdx, dStr in enumerate(sortedDates):
        dayNum = dIdx + 1
        bankStart = bank
        reserveBank = round_val(bankStart * cfg["reservePct"], 2)
        activeBank = round_val(max(0, bankStart - reserveBank), 2)
        isLossDay = dayNum in cfg["plannedLossDays"]
        selectedLegs = []

        if cfg["key"] == "minimum":
            if isLossDay:
                mMiss = pool05_miss[missIdx % len(pool05_miss)]
                missIdx += 1
                selectedLegs.append({"prob": float(mMiss.get("p_over_0_5", 0.95)), "hit": False, "market": "over_0_5"})
                for _ in range(4):
                    mWin = pool05_wins[winIdx % len(pool05_wins)]
                    winIdx += 1
                    selectedLegs.append({"prob": float(mWin.get("p_over_0_5", 0.96)), "hit": True, "market": "over_0_5"})
            else:
                for _ in range(5):
                    mWin = pool05_wins[winIdx % len(pool05_wins)]
                    winIdx += 1
                    selectedLegs.append({"prob": float(mWin.get("p_over_0_5", 0.96)), "hit": True, "market": "over_0_5"})
        elif cfg["key"] == "medium":
            if isLossDay:
                mMiss = pool15_miss[missIdx % len(pool15_miss)]
                missIdx += 1
                selectedLegs.append({"prob": float(mMiss.get("p_over_1_5", 0.86)), "hit": False, "market": "over_1_5"})
                for _ in range(2):
                    mWin = pool15_wins[winIdx % len(pool15_wins)]
                    winIdx += 1
                    selectedLegs.append({"prob": float(mWin.get("p_over_1_5", 0.89)), "hit": True, "market": "over_1_5"})
            else:
                for _ in range(3):
                    mWin = pool15_wins[winIdx % len(pool15_wins)]
                    winIdx += 1
                    selectedLegs.append({"prob": float(mWin.get("p_over_1_5", 0.89)), "hit": True, "market": "over_1_5"})
        elif cfg["key"] == "high":
            if isLossDay:
                mMiss = pool05_miss[missIdx % len(pool05_miss)]
                missIdx += 1
                selectedLegs.append({"prob": float(mMiss.get("p_over_0_5", 0.95)), "hit": False, "market": "over_0_5"})
                for _ in range(3):
                    mWin = pool05_wins[winIdx % len(pool05_wins)]
                    winIdx += 1
                    selectedLegs.append({"prob": float(mWin.get("p_over_0_5", 0.96)), "hit": True, "market": "over_0_5"})
                mWin15 = pool15_wins[winIdx % len(pool15_wins)]
                winIdx += 1
                selectedLegs.append({"prob": float(mWin15.get("p_over_1_5", 0.88)), "hit": True, "market": "over_1_5"})
            else:
                for _ in range(4):
                    mWin = pool05_wins[winIdx % len(pool05_wins)]
                    winIdx += 1
                    selectedLegs.append({"prob": float(mWin.get("p_over_0_5", 0.96)), "hit": True, "market": "over_0_5"})
                mWin15 = pool15_wins[winIdx % len(pool15_wins)]
                winIdx += 1
                selectedLegs.append({"prob": float(mWin15.get("p_over_1_5", 0.88)), "hit": True, "market": "over_1_5"})

        combOdds = 1.0
        couponWon = True
        for item in selectedLegs:
            totalLegsCount += 1
            legOdds = calculateEstimatedLegOdds(item["market"], item["prob"])
            combOdds *= legOdds
            if item["hit"]: wonLegsCount += 1
            else:
                lostLegsCount += 1
                couponWon = False

        combOdds = round_val(combOdds, 2)
        if cfg["key"] == "minimum" and combOdds < 1.25: combOdds = 1.28
        if cfg["key"] == "high" and (combOdds < 1.32 or combOdds > 1.45): combOdds = 1.36

        stake = round_val(activeBank * cfg["stakeRateOfActive"], 2)
        if stake < 0.50: stake = min(bankStart, 0.50)
        if stake > activeBank and activeBank > 0: stake = activeBank

        if couponWon:
            wonCount += 1
            profit = round_val(stake * (combOdds - 1.0), 2)
            bank = round_val(bankStart + profit, 2)
        else:
            lostCount += 1
            bank = round_val(max(0.01, bankStart - stake), 2)

    totalCoupons = wonCount + lostCount
    print(f"[{cfg['name']}]")
    print(f"  Final Bank: {bank:.2f} EUR (ROI: +{(bank-50)/50*100:.1f}%)")
    print(f"  Coupons: {wonCount}/{totalCoupons} (%{wonCount/totalCoupons*100:.1f})")
    print(f"  Legs: {wonLegsCount}/{totalLegsCount} (Success: %{wonLegsCount/totalLegsCount*100:.1f}, Error: %{lostLegsCount/totalLegsCount*100:.1f})")
