def two_sum(nums, target):
    seen = {}

    for index, num in enumerate(nums):
        needed = target - num

        if needed in seen:
            return (seen[needed], index)

        seen[num] = index
    return


if __name__ == "__main__":
    nums = [7, 2, 11, 15]
    target = 9
    result = two_sum(nums, target)

    print("input :", nums)
    print("target :", target)
    print("result :", result)
