def fetch_tle(identifier: str) -> Tuple[str, str]:
    """Fetch TLE lines from Celestrak using catalog number or common name."""
    params = {"FORMAT": "TLE", "CATNR": identifier}
    # If identifier is not numeric, use NAME param instead
    if not identifier.isdigit():
        params = {"FORMAT": "TLE", "NAME": identifier}
    resp = requests.get(CELESTRAK_URL, params=params, timeout=10)
    text = resp.text.strip()
    lines = text.splitlines()
    if resp.status_code != 200 or len(lines) < 2:
        raise TLEFetchError(f"Failed to retrieve TLE for {identifier}: {resp.text}")

    # Many Celestrak endpoints return 3 lines: NAME, L1, L2.
    # We want the LAST TWO lines, which are always the actual TLE.
    if len(lines) >= 3:
        l1, l2 = lines[-2].strip(), lines[-1].strip()
    else:  # exactly 2 lines: already L1, L2
        l1, l2 = lines[0].strip(), lines[1].strip()

    return l1, l2