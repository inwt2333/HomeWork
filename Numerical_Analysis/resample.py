def resample(series, tgt_length):
    """
    对 series 输入序列进行重采样，使得重采样后的长度等于tgt_length
    注:请使用分段线性插值方法，假设 x0，x1，...依次是0,1...

    Args:
    series(List[float]):一维实数序列(长度大于 0)
    tgt_length(int):期望输出的序列长度(一定大于 0)
    Returns:
    ret(List[float]):重采样后的一维序列
    """
    n = len(series)
    if tgt_length <= 0 or n == 0:
        raise ValueError("series 长度和 tgt_length 必须大于 0")
    if n == 1:
        return [series[0]] * tgt_length
    if tgt_length == 1:
        return [series[0]]

    L = tgt_length
    scale = (n - 1) / (L - 1)
    out = []
    for j in range(L):
        pos = j * scale
        left = int(pos)
        right = min(left + 1, n - 1)
        t = pos - left
        v = (1 - t) * series[left] + t * series[right]
        out.append(v)
    return out


print(resample([1, 3, 4, 2],3))