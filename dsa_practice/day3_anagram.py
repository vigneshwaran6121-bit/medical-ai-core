def anagram(s: str, t: str):
    if len(s) != len(t):
        return False

    count_s = {}
    count_t = {}

    for chr in s:
        count_s[chr] = count_s.get(chr, 0) + 1
    for chr in t:
        count_t[chr] = count_t.get(chr, 0) + 1

    return count_s == count_t


print(anagram("car", "arc"))
