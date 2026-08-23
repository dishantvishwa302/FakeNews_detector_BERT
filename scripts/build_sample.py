"""Build data/sample_titles.csv — 1,600 synthetic headlines, 800 real / 800 fake.

These titles exist so the repo runs offline. They are a style contrast
(news-agency vs clickbait), not a claim about the real world.
See data/DATA.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_titles.csv"
N_EACH = 800


def real_titles() -> list[str]:
    agencies = [
        "The Federal Reserve",
        "The European Central Bank",
        "The Bank of England",
        "The IMF",
        "The World Health Organization",
        "NASA",
        "The International Energy Agency",
        "India's finance ministry",
        "Japan's trade ministry",
        "Germany's labor office",
        "Canada's statistical agency",
        "Australia's weather bureau",
        "South Korea's trade office",
        "Brazil's central bank",
        "The OECD",
        "Singapore's monetary authority",
        "France's health ministry",
        "Mexico's energy regulator",
        "Norway's oil fund",
        "The United Nations",
    ]
    events = [
        "holds interest rates after a two-day meeting",
        "revises quarterly inflation figures",
        "reports a modest rebound in factory output",
        "confirms jobless claims below last week's estimate",
        "announces a logistics agreement with a partner country",
        "delays a satellite launch window by two weeks",
        "approves stricter vehicle-emission testing rules",
        "raises the reserve requirement for commercial banks",
        "cuts its 2026 oil-demand estimate by a small margin",
        "publishes a three-year drought-monitoring plan",
        "says merchandise trade volumes ticked up",
        "reports a decline in measles cases in East Africa",
        "updates guidance on hospital staffing ratios",
        "extends a student-visa pilot for one year",
        "issues a heat-wave warning for inland regions",
        "confirms a recall of one packaged-salad batch",
        "holds talks on a grain-export corridor",
        "approves funding for rural-road repairs",
        "reports core inflation holding near two percent",
        "says weekly jobless claims fell below the four-week average",
        "announces a timetable for rail-safety upgrades",
        "revises preliminary GDP growth to 1.8 percent",
        "delays a tourism-fee rollout by one quarter",
        "reports container traffic up from a year earlier",
        "confirms a pause in an emergency lending facility",
    ]
    when = ["on Tuesday", "on Wednesday", "in its monthly note", "in a joint statement"]

    extras = [
        "Oil prices fall as U.S. inventories rise more than expected",
        "India and Japan sign a defense logistics sharing agreement",
        "UK inflation cools to 2.3 percent in the latest reading",
        "Canada adds 28,000 jobs in the monthly labor report",
        "South Korea exports rise 4 percent from a year earlier",
        "Germany factory orders fall 1.1 percent in May",
        "Mexico's peso strengthens after the central bank holds rates",
        "Sweden keeps its policy rate unchanged at the July meeting",
        "Spain's tourism arrivals exceed the pre-pandemic monthly high",
        "Indonesia raises the reserve requirement for commercial banks",
        "New Zealand dairy prices ease at the fortnightly auction",
        "Turkey inflation slows from the prior month's peak",
        "Italy industrial production is unchanged in June",
        "U.S. weekly jobless claims fall below the four-week average",
        "Tokyo core inflation holds at 2.1 percent",
        "Sydney records above-average rainfall for the month",
        "Chennai port reports faster turnaround for container ships",
        "Mumbai monsoon rains are near the long-term average",
        "Bengaluru airport handles a record day of domestic flights",
        "IMF trims global growth forecast to 3.1 percent",
        "WTO reports a small rebound in merchandise trade volumes",
        "FAO food-price index eases for a second month",
        "World Bank approves a loan for rural-road repairs",
        "G20 finance deputies meet to review debt-restructuring terms",
        "Chennai-Delhi daytime flights report a weekday high",
        "London Heathrow passenger numbers rise in July",
        "Paris air quality improves after a weekend driving limit",
        "Seoul apartment prices stabilize after a year of declines",
        "Toronto housing starts fall from a strong May reading",
        "California issues a drought update for Central Valley farms",
    ]

    titles = list(extras)
    for agency in agencies:
        for event in events:
            for w in when:
                titles.append(f"{agency} {event} {w}")
    return _unique(titles)


def fake_titles() -> list[str]:
    openers = [
        "You won't believe",
        "Doctors are stunned as",
        "Secret documents prove",
        "They don't want you to know",
        "Breaking: insiders confirm",
        "Shocking leak reveals",
        "Scientists are terrified because",
        "A whistleblower just admitted",
        "Hidden cameras caught",
        "This one kitchen trick shows",
        "Miracle report claims",
        "Anonymous files show",
        "Viral doctor says",
        "Leaked patent shows",
        "Tabloid exclusive:",
        "Unverified blog insists",
        "A private telegram channel claims",
        "This screenshot 'proves'",
        "Late-night infomercial reveals",
    ]
    claims = [
        "this spice reverses aging overnight",
        "the moon landing was filmed in a hangar",
        "banks are replacing cash with a tracking chip",
        "a celebrity sold the election for a private island",
        "weather satellites are spraying the sky on purpose",
        "hospitals hide a 24-hour cure for the common cold",
        "your phone listens even when it is off",
        "a lost city was found under a shopping mall",
        "school textbooks were rewritten by a secret club",
        "drinking ice water burns fat while you sleep",
        "birds are not real and the proof is in a memo",
        "the stock market is a video game for five families",
        "gravity was mis-measured to hide free energy",
        "a president's double attends all night events",
        "the pyramids are still used as power plants",
        "lottery numbers are chosen in a basement in Zurich",
        "all clocks were moved to hide a missing day in 1973",
        "your toaster is part of a nationwide listening grid",
        "a secret second moon controls interest rates",
        "chewing gum is a government tracking program",
        "cats are running a parallel internet",
        "the earth's core is hollow and there is a mall inside",
        "Wi-Fi routers were designed to erase childhood memories",
        "elevators go to a secret floor where elections are decided",
        "a grocery barcode summons a tax auditor to your home",
        "rain is now optional if you buy this $19.99 whistle",
        "your pillow records conversations for a sleep clinic",
        "the sun will go dark for three days next month",
        "a free-energy device was banned after it powered a town",
        "time travelers keep stealing socks, a laundry study claims",
        "a forgotten vitamin lets you breathe underwater",
        "ocean plastic is a cover for mining gold on the seafloor",
        "your microwave opens a portal if you run it empty",
        "fish learned English and are waiting to testify",
        "a desert bunker holds next year's sports scores",
        "onions make you cry because they contain confession gas",
        "a $1 coin from 1999 opens every parking gate on earth",
        "the night sky is a screensaver while satellites are serviced",
        "your calendar app deletes days that are inconvenient for banks",
        "a viral lemon-peel wrap cures debt and knee pain",
    ]
    extras = [
        "Miracle tea melts 30 pounds in four days, nutritionists furious",
        "Alien craft lands behind a stadium, officials seize all phones",
        "This mom discovered a $2 cure Big Pharma tried to bury",
        "World leaders meet in a volcano to reset the calendar",
        "Man lives to 190 after eating only moonlight and salt",
        "Celebrity clone army spotted leaving a desert studio",
        "Hospitals charge extra because the cure is a common herb",
        "A mayor banned shadows after they started collecting data",
        "You can charge a phone by shouting at it, leaked patent shows",
        "The weekend was invented to hide a third workday",
        "Dentists add radio chips to fillings, anonymous review claims",
        "A supermarket loyalty card unlocks a private army",
        "The equator is a painted line maintained by contractors",
        "Sleep is optional after this 11-second audio file",
        "Airports scan your dreams at security, whistleblower video",
        "Streetlights dim when they detect unapproved thoughts",
        "Your fridge light stays on to photograph leftovers for insurers",
        "People born in March already know next week's headlines",
        "The color blue was patented and will soon require a license",
        "Clouds are rented by the hour in three major cities",
        "Doctors hate this one stone that replaces all medicine",
        "Your printer jams on purpose to sell ink to a cartel",
        "Sidewalk cracks are a barcode for delivery drones",
        "Yawning is a software update for the brain",
        "The hold music at banks is a hypnosis track",
        "Your first name was sold to an ad network at birth",
        "Traffic cones are a voting system for the asphalt lobby",
        "A stadium wave can change a state border if it is big enough",
        "Doctors banned walking because the ground is a treadmill",
        "A leaked memo says clouds are a subscription service now",
        "Your smart speaker files a report when you hum off-key",
        "They hid a second Atlantic ocean under Kansas",
        "Houseplants vote in local elections after midnight",
        "A secret elevator in libraries goes to next year's bestsellers",
        "They bottled fog and sold it as extra weekend",
        "Your spam folder is the only accurate newspaper",
        "They replaced the horizon with a billboard",
        "A secret second alphabet will be mandatory in March",
        "They sold the weekend to a streaming service",
        "Your autocorrect is a foreign correspondent",
        "A park sprinkler is a weather-control test on Tuesdays",
        "The 'check engine' light is a political message",
        "A 'wrong number' call is how the weather is decided",
        "Glue on envelopes is a personality test for the postal service",
        "A park bench records how long you sit and sells it to banks",
        "Windows that face west are taxed at a higher rate",
        "Your sneeze pattern is a password you never agreed to",
        "A bridge closed because it was listening",
        "The 'close door' button in elevators emails your boss",
        "Soap commercials contain a second plot if you play them backwards",
    ]

    titles = list(extras)
    for opener in openers:
        for claim in claims:
            titles.append(f"{opener} {claim}")
    return _unique(titles)


def _unique(titles: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in titles:
        t = " ".join(raw.split())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main() -> None:
    real = real_titles()
    fake = fake_titles()
    if len(real) < N_EACH or len(fake) < N_EACH:
        raise SystemExit(f"Need {N_EACH} each, got real={len(real)} fake={len(fake)}")

    rows = [{"title": t, "label": 0} for t in real[:N_EACH]] + [
        {"title": t, "label": 1} for t in fake[:N_EACH]
    ]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df)} rows)")
    print(df["label"].value_counts().sort_index().to_string())
    print("label 0 = real, label 1 = fake")


if __name__ == "__main__":
    main()
