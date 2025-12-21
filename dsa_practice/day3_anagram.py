def is_anagram(s: str, t: str):
    if len(s) != len(t):
        return False
    count_s = {}
    count_t = {}

    for i in range(len(s)):
        count_s[s[i]] = count_s.get(s[i], 0) + 1
        count_t[t[i]] = count_t.get(t[i], 0) + 1

    return count_s == count_t


if __name__ == "__main__":
    print(f"test 1 (anagram, nagaram): {is_anagram("anagram", "nagaram")}")
    print(f"test2 (rat,car): {is_anagram('rat', 'car')}")
