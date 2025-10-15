"""
获取所有主力合约的60分钟分时行情数据
使用akshare接口：
- match_main_contract: 获取主力合约列表
- futures_zh_minute_sina: 获取分时行情数据
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

class MainContractDataFetcher:
    def __init__(self, output_dir='./data/60min_main_contracts', filter_varieties=None):
        """
        参数:
        - output_dir: 输出目录
        - filter_varieties: 需要筛选的品种代码列表，如 ['V', 'M', 'C']，None表示不筛选
        """
        self.output_dir = output_dir
        self.filter_varieties = filter_varieties
        os.makedirs(output_dir, exist_ok=True)

        # 策略参数
        self.strategy_config = {
            'hhv_entry_period': 20,  # 入场HHV周期
            'llv_stop_period': 10,   # LLV止损周期
            'atr_period': 20,
            'stop_loss_atr': 2,
            'take_profit_atr': 4.0,
            'max_loss_per_trade': 500,  # 每笔交易最大亏损
        }

        # 合约乘数映射表
        self.contract_multipliers = {
            'V': 5,      # PVC
            'M': 10,     # 豆粕
            'C': 10,     # 玉米
            'TA': 5,     # PTA
            'CF': 5,     # 棉花
            'RB': 10,    # 螺纹钢
            'B': 10,     # 豆二
        }

        # 存储所有信号
        self.signals = []

    def extract_variety_code(self, symbol):
        """
        从合约代码中提取品种代码
        例如: 'V2501' -> 'V', 'M2501' -> 'M', 'RB2501' -> 'RB'
        """
        import re
        match = re.match(r'^([a-zA-Z]+)', str(symbol))
        if match:
            return match.group(1).upper()
        return None

    def get_main_contracts(self):
        """获取所有主力合约（从各个交易所）"""
        try:
            print("📊 正在获取主力合约列表...")

            all_contracts = []

            # 各个交易所
            exchanges = {
                'dce': '大连商品交易所',
                'czce': '郑州商品交易所',
                'shfe': '上海期货交易所',
                'gfex': '广州期货交易所',
                'cffex': '中金所'
            }

            for exchange_code, exchange_name in exchanges.items():
                try:
                    print(f"  🔄 获取 {exchange_name} ({exchange_code})...")

                    # 获取该交易所的主力合约
                    contracts_str = ak.match_main_contract(symbol=exchange_code)

                    if contracts_str and isinstance(contracts_str, str):
                        # 解析字符串为列表
                        contracts = [c.strip() for c in contracts_str.split(',') if c.strip()]

                        if contracts:
                            print(f"  ✅ {exchange_name}: 获取到 {len(contracts)} 个主力合约")
                            print(f"     {', '.join(contracts[:5])}{'...' if len(contracts) > 5 else ''}")
                            all_contracts.extend(contracts)
                        else:
                            print(f"  ⚠️ {exchange_name}: 解析结果为空")
                    else:
                        print(f"  ⚠️ {exchange_name}: 未获取到数据")

                except Exception as e:
                    print(f"  ❌ {exchange_name}: 获取失败 - {str(e)}")
                    continue

            if all_contracts:
                # 去重
                all_contracts = list(set(all_contracts))
                print(f"\n✅ 总共获取 {len(all_contracts)} 个主力合约（已去重）")
                return all_contracts
            else:
                print("❌ 未获取到任何主力合约")
                return None

        except Exception as e:
            print(f"❌ 获取主力合约失败: {str(e)}")
            return None

    def fetch_60min_data(self, symbol, contract_name=''):
        """
        获取指定合约的60分钟分时数据

        参数:
        - symbol: 合约代码，如 'M2501'
        - contract_name: 合约名称（用于日志显示）
        """
        try:
            print(f"  🔄 获取 {contract_name or symbol} 的60分钟数据...")

            # 获取分时数据（新浪接口）
            # period参数: 5, 15, 30, 60 (分钟)
            df = ak.futures_zh_minute_sina(
                symbol=symbol,
                period='60'
            )

            if df is not None and len(df) > 0:
                print(f"  ✅ {symbol}: 获取到 {len(df)} 条数据")
                return df
            else:
                print(f"  ⚠️ {symbol}: 未获取到数据")
                return None

        except Exception as e:
            print(f"  ❌ {symbol} 获取失败: {str(e)}")
            return None

    def save_to_csv(self, df, filename):
        """保存数据到CSV文件"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  💾 已保存: {filepath}")
            return True
        except Exception as e:
            print(f"  ❌ 保存失败: {str(e)}")
            return False

    def calculate_position_size(self, symbol, entry_price, stop_loss):
        """
        根据风险金额计算仓位大小
        公式：手数 = 最大亏损金额 / (止损点数 × 合约乘数)
        """
        # 获取品种代码
        variety_code = self.extract_variety_code(symbol)
        if not variety_code or variety_code not in self.contract_multipliers:
            return 0, 0, f"未知品种: {variety_code}"

        # 获取合约乘数
        multiplier = self.contract_multipliers[variety_code]

        # 计算止损点数
        stop_loss_points = abs(entry_price - stop_loss)

        if stop_loss_points <= 0:
            return 0, multiplier, "止损点数无效"

        # 计算单手风险
        risk_per_lot = stop_loss_points * multiplier

        # 计算手数
        max_loss = self.strategy_config['max_loss_per_trade']
        position_size = int(max_loss / risk_per_lot)

        # 如果计算出的手数小于1，检查1手风险是否超过最大亏损
        if position_size < 1:
            if risk_per_lot > max_loss:
                return 0, multiplier, f"1手风险({risk_per_lot:.2f}元)超过最大亏损({max_loss}元)"
            else:
                position_size = 1

        # 最多10手（防止异常）
        position_size = min(position_size, 10)

        return position_size, multiplier, "OK"

    def calculate_atr(self, df, period):
        """
        计算 ATR (Average True Range) - 纯 Python 实现

        ATR 计算公式：
        1. True Range (TR) = max(high - low, abs(high - prev_close), abs(low - prev_close))
        2. ATR = TR 的指数移动平均
        """
        # 计算 True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # 计算 ATR (使用指数移动平均)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_indicators(self, df):
        """计算技术指标"""
        # ATR
        df['ATR'] = self.calculate_atr(df, self.strategy_config['atr_period'])

        # HHV - 最高价的最高值
        df['HHV_entry'] = df['high'].rolling(window=self.strategy_config['hhv_entry_period']).max()

        # LLV - 最低价的最低值
        df['LLV_entry'] = df['low'].rolling(window=self.strategy_config['hhv_entry_period']).min()

        # LLV止损周期
        df['LLV_trail_stop'] = df['low'].rolling(window=self.strategy_config['llv_stop_period']).min()
        # HHV止损周期
        df['HHV_trail_stop'] = df['high'].rolling(window=self.strategy_config['llv_stop_period']).max()

        return df

    def check_signal(self, df, symbol):
        """检查最后3根K线是否有交易信号"""
        if len(df) < 4:
            return []

        signals = []

        # 检查最后3根K线
        for i in range(-3, 0):
            idx = i  # -3, -2, -1
            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            # 检查必要指标
            if pd.isna(prev['HHV_entry']) or pd.isna(current['ATR']):
                continue

            # 多头信号：最高价突破前一根K线的HHV
            if current['high'] > prev['HHV_entry']:
                entry_price = max(prev['HHV_entry'], current['open'])
                stop_loss = entry_price - current['ATR'] * self.strategy_config['stop_loss_atr']
                take_profit = entry_price + current['ATR'] * self.strategy_config['take_profit_atr']

                # 计算仓位大小
                position_size, multiplier, msg = self.calculate_position_size(symbol, entry_price, stop_loss)

                signal = {
                    'contract': symbol,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal_time': current['datetime'],
                    'current_price': current['close'],
                    'hhv_value': prev['HHV_entry'],
                    'atr_value': current['ATR'],
                    'bar_index': f"倒数第{abs(idx)}根",
                    'position_size': position_size,
                    'multiplier': multiplier,
                    'position_msg': msg,
                }
                signals.append(signal)

            # 空头信号：最低价突破前一根K线的LLV
            if current['low'] < prev['LLV_entry']:
                entry_price = min(prev['LLV_entry'], current['open'])
                stop_loss = entry_price + current['ATR'] * self.strategy_config['stop_loss_atr']
                take_profit = entry_price - current['ATR'] * self.strategy_config['take_profit_atr']

                # 计算仓位大小
                position_size, multiplier, msg = self.calculate_position_size(symbol, entry_price, stop_loss)

                signal = {
                    'contract': symbol,
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal_time': current['datetime'],
                    'current_price': current['close'],
                    'llv_value': prev['LLV_entry'],
                    'atr_value': current['ATR'],
                    'bar_index': f"倒数第{abs(idx)}根",
                    'position_size': position_size,
                    'multiplier': multiplier,
                    'position_msg': msg,
                }
                signals.append(signal)

        return signals

    def prepare_data_for_signal(self, df):
        """准备数据用于信号检查"""
        # 统一列名
        column_mapping = {
            'date': 'datetime',
            'Date': 'datetime',
            'TIME': 'datetime',
            'time': 'datetime',
            '时间': 'datetime',
            'OPEN': 'open',
            'Open': 'open',
            '开盘价': 'open',
            'HIGH': 'high',
            'High': 'high',
            '最高价': 'high',
            'LOW': 'low',
            'Low': 'low',
            '最低价': 'low',
            'CLOSE': 'close',
            'Close': 'close',
            '收盘价': 'close',
        }

        # 应用列名映射
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return None

        # 数据清洗
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        if len(df) < self.strategy_config['hhv_entry_period'] + 10:
            return None

        # 计算技术指标
        df = self.calculate_indicators(df)
        return df

    def run(self, delay=0.5, max_contracts=None):
        """
        主运行函数

        参数:
        - delay: 每次请求之间的延迟（秒），避免请求过快
        - max_contracts: 最多获取多少个合约（用于测试，None表示全部）
        """
        print("=" * 80)
        print("主力合约60分钟数据获取程序")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 1. 获取主力合约列表
        main_contracts = self.get_main_contracts()

        if main_contracts is None or len(main_contracts) == 0:
            print("❌ 无法继续：未获取到主力合约列表")
            return

        # 显示筛选信息并立即过滤
        if self.filter_varieties:
            print(f"\n🔍 品种筛选已启用，只获取以下品种的数据:")
            print(f"   {', '.join(self.filter_varieties)}")

            # 立即过滤出需要的合约
            filtered_contracts = []
            for symbol in main_contracts:
                variety_code = self.extract_variety_code(symbol)
                if variety_code in self.filter_varieties:
                    filtered_contracts.append(symbol)

            print(f"\n✅ 从 {len(main_contracts)} 个主力合约中筛选出 {len(filtered_contracts)} 个:")
            print(f"   {', '.join(filtered_contracts)}")

            main_contracts = filtered_contracts

            if len(main_contracts) == 0:
                print("❌ 筛选后无可用合约")
                return
        else:
            print(f"\n📋 未设置筛选，将获取所有主力合约数据")

        # 2. 遍历每个主力合约，获取60分钟数据
        print(f"\n{'=' * 80}")
        print("开始获取60分钟分时数据")
        print(f"{'=' * 80}\n")

        success_count = 0
        fail_count = 0

        # 限制合约数量（用于测试）
        if max_contracts:
            main_contracts = main_contracts[:max_contracts]
            print(f"⚠️ 测试模式：仅获取前 {max_contracts} 个合约\n")

        total_contracts = len(main_contracts)

        for idx, symbol in enumerate(main_contracts):
            print(f"[{idx + 1}/{total_contracts}] {symbol}")

            # 获取60分钟数据
            df = self.fetch_60min_data(symbol, contract_name='')

            if df is not None and len(df) > 0:
                # 保存数据
                filename = f"{symbol}_60min.csv"
                if self.save_to_csv(df, filename):
                    success_count += 1

                    # 检查交易信号（最后3根K线）
                    print(f"  🔍 检查最后3根K线的交易信号...")
                    df_prepared = self.prepare_data_for_signal(df.copy())
                    if df_prepared is not None:
                        found_signals = self.check_signal(df_prepared, symbol)
                        if found_signals:
                            self.signals.extend(found_signals)
                            print(f"  🚨 发现 {len(found_signals)} 个信号！")
                        else:
                            print(f"  ✓ 无交易信号")
                    else:
                        print(f"  ⚠️ 数据处理失败，无法检查信号")
                else:
                    fail_count += 1
            else:
                fail_count += 1

            # 延迟，避免请求过快
            if idx < total_contracts - 1:
                time.sleep(delay)

            print()  # 空行分隔

        # 3. 汇总统计
        print(f"\n{'=' * 80}")
        print("数据获取完成")
        print(f"{'=' * 80}")
        processed_count = success_count + fail_count
        print(f"总合约数: {total_contracts}")
        print(f"处理合约数: {processed_count}")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        if processed_count > 0:
            print(f"成功率: {success_count / processed_count * 100:.2f}%")
        print(f"数据保存目录: {os.path.abspath(self.output_dir)}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 4. 显示交易信号
        print(f"\n{'=' * 80}")
        print("🎯 HHV突破交易信号汇总")
        print(f"{'=' * 80}")
        print(f"策略参数: HHV({self.strategy_config['hhv_entry_period']}) / "
              f"ATR({self.strategy_config['atr_period']}) / "
              f"止损({self.strategy_config['stop_loss_atr']}ATR) / "
              f"止盈({self.strategy_config['take_profit_atr']}ATR)")
        print("=" * 80)

        if self.signals:
            print(f"\n🚨 共发现 {len(self.signals)} 个交易信号：\n")

            for idx, signal in enumerate(self.signals, 1):
                print(f"【信号 {idx}】")
                print(f"  合约名称: {signal['contract']}")
                print(f"  K线位置: {signal.get('bar_index', '最新')}")
                print(f"  开仓方向: {signal['direction']}")
                print(f"  开仓价格: {signal['entry_price']:.2f}")
                print(f"  止损价格: {signal['stop_loss']:.2f}")
                print(f"  止盈价格: {signal['take_profit']:.2f}")

                # 仓位信息
                position_size = signal.get('position_size', 0)
                multiplier = signal.get('multiplier', 0)
                position_msg = signal.get('position_msg', 'N/A')

                if position_size > 0:
                    print(f"  开仓数量: {position_size} 手 (合约乘数: {multiplier})")
                    total_risk = abs(signal['entry_price'] - signal['stop_loss']) * multiplier * position_size
                    print(f"  预计风险: {total_risk:.2f} 元")
                else:
                    print(f"  开仓数量: 无法开仓 ({position_msg})")

                print(f"  当前价格: {signal['current_price']:.2f}")

                # 当前价格优势分析
                if signal['direction'] == 'LONG':
                    price_diff = signal['current_price'] - signal['entry_price']
                    if price_diff > 0:
                        advantage = price_diff * multiplier * position_size if position_size > 0 else 0
                        print(f"  价格优势: 当前价格高于开仓价 {price_diff:.2f} 点 (不利，建议等待回调)" +
                              (f" 若立即开仓将多付 {advantage:.2f} 元" if advantage > 0 else ""))
                    elif price_diff < 0:
                        advantage = abs(price_diff) * multiplier * position_size if position_size > 0 else 0
                        print(f"  价格优势: 当前价格低于开仓价 {abs(price_diff):.2f} 点 (有利)" +
                              (f" 可节省 {advantage:.2f} 元" if advantage > 0 else ""))
                    else:
                        print(f"  价格优势: 当前价格等于开仓价 (中性)")
                else:  # SHORT
                    price_diff = signal['entry_price'] - signal['current_price']
                    if price_diff > 0:
                        advantage = price_diff * multiplier * position_size if position_size > 0 else 0
                        print(f"  价格优势: 当前价格低于开仓价 {price_diff:.2f} 点 (不利，建议等待反弹)" +
                              (f" 若立即开仓将多付 {advantage:.2f} 元" if advantage > 0 else ""))
                    elif price_diff < 0:
                        advantage = abs(price_diff) * multiplier * position_size if position_size > 0 else 0
                        print(f"  价格优势: 当前价格高于开仓价 {abs(price_diff):.2f} 点 (有利)" +
                              (f" 可节省 {advantage:.2f} 元" if advantage > 0 else ""))
                    else:
                        print(f"  价格优势: 当前价格等于开仓价 (中性)")

                print(f"  信号时间: {signal['signal_time']}")

                if signal['direction'] == 'LONG':
                    print(f"  突破值: HHV({self.strategy_config['hhv_entry_period']}) = {signal.get('hhv_value', 0):.2f}")
                    risk = signal['entry_price'] - signal['stop_loss']
                    reward = signal['take_profit'] - signal['entry_price']
                else:
                    print(f"  突破值: LLV({self.strategy_config['hhv_entry_period']}) = {signal.get('llv_value', 0):.2f}")
                    risk = signal['stop_loss'] - signal['entry_price']
                    reward = signal['entry_price'] - signal['take_profit']

                print(f"  ATR值: {signal['atr_value']:.2f}")
                if risk > 0:
                    print(f"  风险回报比: 1:{reward/risk:.2f}")
                print()
        else:
            print("\n📭 当前无交易信号\n")

        print("=" * 80)


def main():
    """主函数"""
    # 定义需要筛选的品种代码
    # PVC(V), 豆粕(M), 玉米(C), PTA(TA), 棉花(CF), 螺纹钢(RB), 豆二(B)
    target_varieties = ['V', 'M', 'C', 'TA', 'CF', 'RB', 'B']

    print(f"🎯 筛选品种: {', '.join(target_varieties)}")
    print(f"   V=PVC, M=豆粕, C=玉米, TA=PTA, CF=棉花, RB=螺纹钢, B=豆二\n")

    # 创建数据获取器
    fetcher = MainContractDataFetcher(
        output_dir='./data/60min_main_contracts',
        filter_varieties=target_varieties
    )

    # 运行数据获取
    # delay: 每次请求延迟（秒）
    # max_contracts: 测试时可以限制数量，如 max_contracts=5
    fetcher.run(delay=0.5, max_contracts=None)


if __name__ == "__main__":
    main()
