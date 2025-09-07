from mtc import TripleCollocation
import time


def run_main():#跑全球的tc
    start = time.time()
    # 创建 TripleCollocation 类的实例
    tc = TripleCollocation()
    tc.read_npy_files()  # 调用read_npy_files方法加载.npy文件
    # 调用 main 函数
    tc.main()
    end= time.time()
    print('time spent for retrieving one pixel: %.2f seconds'%(end-start))

def run_main_single():
    start = time.time()
    # 创建 TripleCollocation 类的实例
    tc_instance = TripleCollocation()
    tc_instance.read_npy_files()  # 调用read_npy_files方法加载.npy文件
    # 调用 main 函数
    tc_instance.main_single()
    end= time.time()
    print('time spent for retrieving one pixel: %.2f seconds'%(end-start))


def run_single(i, j):
    start = time.time()
    tc_instance = TripleCollocation()
    tc_instance.read_npy_files()  # 调用read_npy_files方法加载.npy文件
    result = tc_instance.single((i, j))
    print(result)
    end= time.time()
    print('time spent for retrieving one pixel: %.2f seconds'%(end-start))

def run_parallel():
    start = time.time()
    tc = TripleCollocation()
    tc.read_npy_files()  # 添加这一行来初始化属性
    tc.parallel(cores=3)
    end = time.time()
    print('time spent for retrieving one pixel: %.2f seconds' % (end - start))

if __name__ == "__main__":
    run_main()
    # run_main_single()
    # run_single(i=360, j=480)
    # run_parallel()