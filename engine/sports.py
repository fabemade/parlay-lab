"""Per-sport configuration.

Numbers here come from well-established betting and sports analytics results (see
METHODOLOGY.md). `kind` picks the scoring model:
  gaussian: high-scoring sports; margins and totals are roughly normal (NFL, NBA, CFB)
  poisson:  low-scoring sports; each team's score is a Poisson/negative-binomial count (MLB, NHL, soccer)
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sport:
    key: str
    name: str
    path: str                       # ESPN API path
    kind: str                       # "gaussian" | "poisson"
    margin_sd: float = 0.0          # gaussian: SD of (actual margin - predicted margin)
    total_sd: float = 0.0           # gaussian: SD of (actual total - predicted total)
    home_adv_prior: float = 0.0     # points (gaussian) or log-rate (poisson) home edge
    dispersion: float = 0.0         # poisson: negative binomial overdispersion (0 = pure Poisson)
    half_life_days: float = 120.0   # recency weighting for team ratings
    history_days: int = 400
    ridge: float = 1.0              # shrinkage of team ratings toward average
    w_model: float = 0.35           # weight of our model vs the de-vigged market
    three_way: bool = False         # soccer: win/draw/win
    draws_go_to_ot: bool = False    # hockey: regulation ties go to OT/shootout
    params: dict = field(default_factory=dict)
    # per-market trust multiplier on w_model (lower where our distribution shape is rougher)
    market_w: dict = field(default_factory=lambda: {"total": 0.35})
    # playoff games are lower scoring (top pitchers/goalies, tighter defense, slower pace)
    postseason_total: float = 1.0


SPORTS: dict[str, Sport] = {s.key: s for s in [
    Sport("nfl", "NFL", "football/nfl", "gaussian", margin_sd=13.2, total_sd=13.0,
          home_adv_prior=1.6, half_life_days=150, history_days=420, ridge=3.0, w_model=0.30),
    Sport("cfb", "College Football", "football/college-football", "gaussian", margin_sd=15.5,
          total_sd=15.5, home_adv_prior=2.8, half_life_days=150, history_days=420, ridge=2.0,
          w_model=0.30, params={"groups": 80}),
    Sport("nba", "NBA", "basketball/nba", "gaussian", margin_sd=12.0, total_sd=17.5,
          home_adv_prior=2.3, half_life_days=90, history_days=300, ridge=4.0, w_model=0.30,
          postseason_total=0.97),
    Sport("ncaab", "College Basketball", "basketball/mens-college-basketball", "gaussian",
          margin_sd=10.8, total_sd=15.0, home_adv_prior=3.2, half_life_days=90,
          history_days=200, ridge=3.0, w_model=0.30, params={"groups": 50, "limit": 400}),
    Sport("wnba", "WNBA", "basketball/wnba", "gaussian", margin_sd=11.5, total_sd=15.0,
          home_adv_prior=2.0, half_life_days=90, history_days=200, ridge=3.0, w_model=0.30,
          postseason_total=0.96),
    Sport("mlb", "MLB", "baseball/mlb", "poisson", home_adv_prior=0.035, dispersion=0.10,
          half_life_days=60, history_days=240, ridge=25.0, w_model=0.30,
          market_w={"spread": 0.6, "total": 0.35}, postseason_total=0.90),
    Sport("nhl", "NHL", "hockey/nhl", "poisson", home_adv_prior=0.05, dispersion=0.02,
          half_life_days=90, history_days=300, ridge=15.0, w_model=0.30, draws_go_to_ot=True,
          market_w={"spread": 0.6, "total": 0.35}, postseason_total=0.93),
] + [
    Sport(f"soccer_{code}", name, f"soccer/{code}", "poisson", home_adv_prior=0.12,
          dispersion=0.02, half_life_days=180, history_days=420, ridge=6.0, w_model=0.30,
          three_way=True)
    for code, name in [
        # top 5 European leagues
        ("eng.1", "Premier League"), ("esp.1", "La Liga"), ("ita.1", "Serie A"),
        ("ger.1", "Bundesliga"), ("fra.1", "Ligue 1"),
        # UEFA club competitions
        ("uefa.champions", "Champions League"), ("uefa.europa", "Europa League"),
        ("uefa.europa.conf", "Conference League"),
        # next tier of European top flights
        ("ned.1", "Eredivisie"), ("por.1", "Primeira Liga"), ("bel.1", "Belgian Pro League"),
        ("tur.1", "Süper Lig"), ("sco.1", "Scottish Premiership"),
        # second divisions: less-watched markets, so prices are softer
        ("eng.2", "Championship"), ("eng.3", "League One"), ("ger.2", "2. Bundesliga"),
        ("esp.2", "LaLiga 2"), ("ita.2", "Serie B"), ("fra.2", "Ligue 2"),
        # Americas and beyond
        ("usa.1", "MLS"), ("mex.1", "Liga MX"), ("bra.1", "Brasileirão"),
        ("arg.1", "Argentine Primera"), ("conmebol.libertadores", "Copa Libertadores"),
        ("ksa.1", "Saudi Pro League"),
    ]
]}

SOCCER = [k for k, s in SPORTS.items() if s.three_way]
