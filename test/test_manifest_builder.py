from scripts.build_exchangeability_manifest import _derive_member_seeds



def test_member_seed_derivation_is_deterministic():
    seeds_a = _derive_member_seeds(12345, 4)
    seeds_b = _derive_member_seeds(12345, 4)
    assert seeds_a == seeds_b
    assert len(seeds_a) == 4
