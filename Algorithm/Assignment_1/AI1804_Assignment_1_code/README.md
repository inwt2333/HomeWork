# 算法作业指南

本仓库包含算法题目的代码框架（`problem/`）与单元测试（`test/`）。请按照本指南完成实现与本地自测。

## 目录结构

- `problem/`
  - 每道题一个 Python 文件，例如 `1_1_Maximum_Subarray.py`。
  - 已提供类与方法签名，请在指定方法内实现。
  - 文件顶部含中英双语的题目简述。
- `test/`
  - 针对每道题的单元测试与性能测试（性能测试需设置环境变量开启）。

## 如何编写代码

请在对应题目的方法中实现，不要修改对外接口名称与返回类型。
- 1-1：`Solution.maxSubArray(self, nums) -> int`
- 1-2：`Solution.findPeakElement(self, nums) -> int`
- 1-3：`Solution.majorityElement(self, nums) -> int`
- 1-4：`Solution.findDuplicate(self, nums) -> int`
- 2-1：`Solution.findKthLargest(self, nums, k) -> int`
- 3-1：`HospitalSystem.assign_patients_to_clinics(self, arrivals, k) -> List[List[int]]`
       `HospitalSystem.merge_clinic_queues(self, queues) -> List[int]`
       `HospitalSystem.process_hospital_queue(self, arrivals, k) -> List[int]`

实现建议：
- 严格遵循 `problem/Problem Description.md`（或 `题目描述.md`）中的约束与复杂度要求。
- 优先保证正确性，其次考虑时间与空间复杂度。
- 不引入第三方依赖，仅使用 Python 标准库。
- 返回值类型需与方法注释一致（测试依赖此约定）。

## 如何测试

在仓库根目录执行下列命令（建议使用 Python 3.9+）。

- 运行全部功能性测试（较快）：

```bash
python -m unittest discover -s test -p "test_*.py" -v
```

- 启用性能测试（用例较大，耗时更长）：

```bash
RUN_PERF=1 python -m unittest discover -s test -p "test_*.py" -v
```

- 只运行某个测试文件：

```bash
python -m unittest test.test_1_1_maximum_subarray -v
```

- 只运行某个测试用例/方法：

```bash
python -m unittest test.test_1_1_maximum_subarray.TestMaximumSubarray.test_examples -v
```

说明：
- 在未实现或返回 `None` 时，部分测试会自动跳过对应用例；当返回实际结果后，将正常断言。
- 性能测试仅打印耗时，不做硬性阈值限制。

## 提交说明

请在完成作业后，提交整个压缩包至canva。

## 联系方式

如有问题，请联系：hrenming13@gmail.com
