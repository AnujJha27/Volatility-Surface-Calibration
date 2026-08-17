from heston_surface import black_scholes_call, implied_vol


def test_implied_vol_round_trip():
    price = black_scholes_call(100, 100, 1, 0.02, 0.25)
    assert abs(implied_vol(price, 100, 100, 1, 0.02) - 0.25) < 1e-6


if __name__ == "__main__":
    test_implied_vol_round_trip()
    print("ok")
