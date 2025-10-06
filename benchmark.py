"""
性能基准测试脚本
比较优化前后的性能差异
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Callable
import sys

# 导入性能工具
try:
    from performance_utils import benchmark_function, PerformanceProfiler, DataFrameOptimizer
    from numba_optimizations import NumbaOptimizedIndicators, NUMBA_AVAILABLE
except ImportError:
    print("请确保performance_utils.py和numba_optimizations.py在同一目录")
    sys.exit(1)


class PerformanceBenchmark:
    """性能基准测试套件"""
    
    def __init__(self, data_size: int = 10000):
        self.data_size = data_size
        self.results = {}
        
        # 生成测试数据
        np.random.seed(42)
        self.prices = 100 + np.cumsum(np.random.randn(data_size) * 0.01)
        self.df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=data_size),
            'Open': self.prices + np.random.randn(data_size) * 0.1,
            'High': self.prices + np.abs(np.random.randn(data_size) * 0.2),
            'Low': self.prices - np.abs(np.random.randn(data_size) * 0.2),
            'Close': self.prices,
            'Volume': np.random.randint(1000000, 10000000, data_size)
        })
    
    def test_data_iteration(self):
        """测试数据迭代方法"""
        print("\n" + "="*70)
        print("测试 1: 数据迭代性能对比")
        print("="*70)
        
        # 方法1: iterrows (慢)
        def method_iterrows():
            total = 0
            for _, row in self.df.iterrows():
                total += row['Close']
            return total
        
        # 方法2: itertuples (中等)
        def method_itertuples():
            total = 0
            for row in self.df.itertuples():
                total += row.Close
            return total
        
        # 方法3: iloc (快)
        def method_iloc():
            total = 0
            for i in range(len(self.df)):
                total += self.df.iloc[i]['Close']
            return total
        
        # 方法4: values (最快)
        def method_values():
            return np.sum(self.df['Close'].values)
        
        # 基准测试
        iterations = 10
        
        print(f"\n迭代 {self.data_size} 行数据，重复 {iterations} 次:")
        
        result1 = benchmark_function(method_iterrows, iterations=iterations)
        print(f"  iterrows:    {result1['mean']*1000:.2f}ms (基准)")
        
        result2 = benchmark_function(method_itertuples, iterations=iterations)
        speedup = result1['mean'] / result2['mean']
        print(f"  itertuples:  {result2['mean']*1000:.2f}ms ({speedup:.1f}x 更快)")
        
        result3 = benchmark_function(method_iloc, iterations=iterations)
        speedup = result1['mean'] / result3['mean']
        print(f"  iloc:        {result3['mean']*1000:.2f}ms ({speedup:.1f}x 更快)")
        
        result4 = benchmark_function(method_values, iterations=iterations)
        speedup = result1['mean'] / result4['mean']
        print(f"  values:      {result4['mean']*1000:.2f}ms ({speedup:.1f}x 更快) ⭐推荐")
        
        self.results['iteration'] = {
            'iterrows': result1['mean'],
            'itertuples': result2['mean'],
            'iloc': result3['mean'],
            'values': result4['mean']
        }
    
    def test_rolling_calculations(self):
        """测试滚动计算性能"""
        print("\n" + "="*70)
        print("测试 2: 滚动均值计算性能对比")
        print("="*70)
        
        window = 20
        
        # 方法1: pandas rolling (慢)
        def method_pandas():
            return self.df['Close'].rolling(window).mean()
        
        # 方法2: numpy (快)
        def method_numpy():
            result = np.empty(len(self.prices))
            result[:window-1] = np.nan
            cumsum = np.cumsum(self.prices)
            cumsum[window:] = cumsum[window:] - cumsum[:-window]
            result[window-1:] = cumsum[window-1:] / window
            return result
        
        # 方法3: Numba (最快，如果可用)
        if NUMBA_AVAILABLE:
            from numba_optimizations import fast_moving_average
            def method_numba():
                return fast_moving_average(self.prices, window)
        
        # 基准测试
        iterations = 100
        
        print(f"\n计算 {window} 期滚动均值，数据大小 {self.data_size}:")
        
        result1 = benchmark_function(method_pandas, iterations=iterations)
        print(f"  pandas rolling: {result1['mean']*1000:.2f}ms (基准)")
        
        result2 = benchmark_function(method_numpy, iterations=iterations)
        speedup = result1['mean'] / result2['mean']
        print(f"  numpy:          {result2['mean']*1000:.2f}ms ({speedup:.1f}x 更快)")
        
        if NUMBA_AVAILABLE:
            result3 = benchmark_function(method_numba, iterations=iterations)
            speedup = result1['mean'] / result3['mean']
            print(f"  numba JIT:      {result3['mean']*1000:.2f}ms ({speedup:.1f}x 更快) ⭐推荐")
    
    def test_memory_optimization(self):
        """测试内存优化"""
        print("\n" + "="*70)
        print("测试 3: 内存优化效果")
        print("="*70)
        
        # 原始DataFrame
        original_memory = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\n原始DataFrame内存使用: {original_memory:.2f} MB")
        
        # 优化DataFrame
        optimized_df = DataFrameOptimizer.optimize_dtypes(self.df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"优化后DataFrame内存使用: {optimized_memory:.2f} MB")
        
        reduction = (1 - optimized_memory / original_memory) * 100
        print(f"内存节省: {reduction:.1f}% ⭐")
        
        # 验证数据一致性
        print(f"\n数据一致性检查:")
        print(f"  Close列最大差异: {np.max(np.abs(self.df['Close'].astype('float32') - self.df['Close'])):.10f}")
    
    def test_concat_vs_list(self):
        """测试concat vs list性能"""
        print("\n" + "="*70)
        print("测试 4: DataFrame拼接方法对比")
        print("="*70)
        
        iterations = 100
        
        # 方法1: 循环中使用concat (极慢)
        def method_concat():
            result = pd.DataFrame()
            for i in range(iterations):
                new_row = pd.DataFrame({'value': [i]})
                result = pd.concat([result, new_row], ignore_index=True)
            return result
        
        # 方法2: 列表缓存 (快)
        def method_list():
            data_list = []
            for i in range(iterations):
                data_list.append({'value': i})
            return pd.DataFrame(data_list)
        
        # 基准测试
        print(f"\n追加 {iterations} 行数据:")
        
        start = time.time()
        method_concat()
        concat_time = time.time() - start
        print(f"  循环concat:  {concat_time*1000:.2f}ms (基准)")
        
        start = time.time()
        method_list()
        list_time = time.time() - start
        speedup = concat_time / list_time
        print(f"  列表缓存:    {list_time*1000:.2f}ms ({speedup:.1f}x 更快) ⭐推荐")
    
    def test_technical_indicators(self):
        """测试技术指标计算性能"""
        print("\n" + "="*70)
        print("测试 5: 技术指标计算性能")
        print("="*70)
        
        # RSI计算
        print(f"\nRSI指标计算 (数据大小: {self.data_size}):")
        
        # Pandas实现 (慢)
        def pandas_rsi(prices, period=14):
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.values
        
        # Numpy实现 (快)
        def numpy_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            rsi = np.zeros(len(prices))
            for i in range(period, len(prices)):
                avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi[i] = 100 - (100 / (1 + rs))
            
            return rsi
        
        iterations = 100
        
        result1 = benchmark_function(pandas_rsi, self.prices, iterations=iterations)
        print(f"  Pandas实现:  {result1['mean']*1000:.2f}ms")
        
        result2 = benchmark_function(numpy_rsi, self.prices, iterations=iterations)
        speedup = result1['mean'] / result2['mean']
        print(f"  Numpy实现:   {result2['mean']*1000:.2f}ms ({speedup:.1f}x 更快)")
        
        if NUMBA_AVAILABLE:
            from numba_optimizations import fast_rsi_calculation
            result3 = benchmark_function(fast_rsi_calculation, self.prices, 14, iterations=iterations)
            speedup = result1['mean'] / result3['mean']
            print(f"  Numba实现:   {result3['mean']*1000:.2f}ms ({speedup:.1f}x 更快) ⭐推荐")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║              量化交易框架性能基准测试                              ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        
        self.test_data_iteration()
        self.test_rolling_calculations()
        self.test_memory_optimization()
        self.test_concat_vs_list()
        self.test_technical_indicators()
        
        # 总结
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        print("\n关键优化建议:")
        print("  1. ✅ 使用 .values 或 numpy 替代 iterrows()")
        print("  2. ✅ 使用列表缓存替代循环中的 pd.concat()")
        print("  3. ✅ 使用 float32 替代 float64 节省50%内存")
        print("  4. ✅ 使用 numpy 向量化操作进行数值计算")
        if NUMBA_AVAILABLE:
            print("  5. ✅ 使用 Numba JIT 进一步加速密集计算")
        else:
            print("  5. ⚠️  安装 Numba 以获得更大性能提升: pip install numba")
        
        print(f"\n测试完成！数据规模: {self.data_size} 条记录")


def quick_benchmark():
    """快速基准测试"""
    print("运行快速性能测试...")
    
    # 小数据集
    print("\n【小数据集测试: 1,000条】")
    benchmark_small = PerformanceBenchmark(data_size=1000)
    benchmark_small.test_data_iteration()
    
    # 中等数据集
    print("\n【中等数据集测试: 10,000条】")
    benchmark_medium = PerformanceBenchmark(data_size=10000)
    benchmark_medium.test_rolling_calculations()
    
    # 大数据集
    print("\n【大数据集测试: 100,000条】")
    benchmark_large = PerformanceBenchmark(data_size=100000)
    benchmark_large.test_memory_optimization()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_benchmark()
    else:
        # 完整测试
        benchmark = PerformanceBenchmark(data_size=10000)
        benchmark.run_all_tests()
