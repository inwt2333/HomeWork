def get_significant_figure(ref, est) -> int:
    """计算实数估计值 est 相对于实数参考值 ref 的有效数字位数

         注：应支持负数情况。【可选】支持 1.5e-3 科学计数法形式的输入

    Args:

        ref (str): 实数参考值的字符串形式

        est (str): 实数估计值的字符串形式

    Returns:

        n (int): 有效数字位数

    """

    def normalize(num_str: str):
    # 将输入的字符串转化为 (sign, digits, exp)
        s = num_str.strip()
        sign = -1 if s[0] == '-' else 1
        if s and s[0] == '-':
            s = s[1:]

        # 处理指数部分
        if 'e' in s:
            base, epart = s.split('e') # epart 可能带正负号
            exp_e = int(epart or '0')
        else:
            base = s
            exp_e = 0

        # 处理小数点部分
        if '.' in base:
            int_part, frac_part = base.split('.')
        else:
            int_part, frac_part = base, ''

        int_part = int_part.lstrip('0') # 去掉整数部分前导零
        digits_raw = int_part + frac_part 
        if digits_raw == '':
            digits_raw = '0'

        # 如果全是 0
        if all(ch == '0' for ch in digits_raw):
            return sign, '0', 0
        
        # 计算指数
        exp = exp_e - len(frac_part)

        # 去除原小数部分前导零
        i = 0
        while i < len(digits_raw) and digits_raw[i] == '0':
            i += 1
        digits = digits_raw[i:]

        return sign, digits, exp

    def compare_digits(d1, e1, d2, e2):
        # 若数量级相差过大，直接返回 0
        if abs((len(d1) + e1) - (len(d2) + e2)) > 1: # 数量级相差超过 1
            return 0
        # 对齐到相同指数
        shift = e1 - e2
        if shift > 0:
            d1 += '0' * shift
            e1 -= shift
        elif shift < 0:
            d2 += '0' * (-shift)
            e2 -= (-shift)

        # print(d1, e1, d2, e2)
        diff = abs(int(d1) - int(d2))
        if diff == 0:
            return min(len(d1), len(d2))
        t = len(str(diff)) - 1 + e2
        # print(t)
        if diff > 5 * (10 ** (len(str(diff)) - 1)): # 四舍五入进位
            t += 1
        m = len(d2) + e2 - 1
        n = m - t
        # print(diff, t, m, n)
        return n
            

    s1, d1, e1 = normalize(ref)
    s2, d2, e2 = normalize(est)

    if s1 != s2: # 符号不一致
        return 0

    # print(s1, d1, e1)
    # print(s2, d2, e2)
    n = max(min(compare_digits(d1, e1, d2, e2), len(d1), len(d2)), 0)
    # print(n)
    return n

if __name__ == "__main__":

    # 可自行在此代码块下面添加更多测试样例
    print(get_significant_figure('1.0e100', '1.0e100')) # 2



