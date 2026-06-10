"""
Step 1.1 — Vanilla European pricer (BSM + Heston engines)
Step 1.2 — Implied vol inverter

Source: Structured Note Pricing Model Technical Reference, §4
"""
import QuantLib as ql
from vollib.black_scholes.implied_volatility import implied_volatility as iv


class VanillaEuropean:
    def __init__(self, S, K, expiry_date, eval_date, option_type='put',
                 r=0.04, q=0.0):
        ql.Settings.instance().evaluationDate = eval_date
        self.spot = ql.QuoteHandle(ql.SimpleQuote(S))
        self.rate_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, r, ql.Actual365Fixed()))
        self.div_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, q, ql.Actual365Fixed()))
        opt_type = ql.Option.Put if option_type == 'put' else ql.Option.Call
        payoff = ql.PlainVanillaPayoff(opt_type, K)
        exercise = ql.EuropeanExercise(expiry_date)
        self.option = ql.VanillaOption(payoff, exercise)

    def price_bsm(self, sigma):
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                ql.Settings.instance().evaluationDate,
                ql.NullCalendar(), sigma, ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(
            self.spot, self.div_ts, self.rate_ts, vol_ts)
        self.option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return self.option.NPV()

    def price_heston(self, v0, kappa, theta, sigma_vov, rho):
        process = ql.HestonProcess(
            self.rate_ts, self.div_ts, self.spot,
            v0, kappa, theta, sigma_vov, rho)
        model = ql.HestonModel(process)
        self.option.setPricingEngine(
            ql.AnalyticHestonEngine(model))
        return self.option.NPV()


def market_iv(market_price, S, K, t_years, r, option_type='p'):
    """Recover BSM implied vol from a market price using vollib."""
    try:
        return iv(market_price, S, K, t_years, r, option_type)
    except Exception:
        return None  # iv inversion failed; usually arb violation
