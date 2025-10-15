"""
期货主力合约交易信号 API 服务
提供 HTTP RESTful API 接口获取交易信号
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from fetch_main_contracts_60min import MainContractDataFetcher
import logging
from datetime import datetime
import traceback
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 默认配置
DEFAULT_CONFIG = {
    'output_dir': './data/60min_main_contracts',
    'filter_varieties': ['V', 'M', 'C', 'TA', 'CF', 'RB', 'B'],
    'delay': 0.5,
    'max_contracts': None
}


@app.route('/', methods=['GET'])
def home():
    """主页 - 返回API文档"""
    return jsonify({
        'service': '期货交易信号API',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/health': '健康检查',
            'GET /api/signals': '获取交易信号',
            'GET /api/varieties': '获取支持的品种列表',
            'GET /api/config': '获取当前配置'
        },
        'example': 'GET /api/signals?varieties=V,M,C'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': 'futures-signal-api',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """
    获取期货交易信号

    查询参数:
    - varieties: 品种代码列表，逗号分隔，如 "V,M,C"，默认为所有配置的品种
    - max_contracts: 最大合约数量，用于测试，默认 None（全部）
    - delay: 请求延迟（秒），默认 0.5

    返回:
    {
        "success": true,
        "timestamp": "2025-01-15T10:30:00",
        "signals": [...],
        "statistics": {...}
    }
    """
    try:
        # 获取查询参数
        varieties_param = request.args.get('varieties', '')
        max_contracts = request.args.get('max_contracts', None)
        delay = float(request.args.get('delay', DEFAULT_CONFIG['delay']))

        # 解析品种列表
        if varieties_param:
            filter_varieties = [v.strip().upper() for v in varieties_param.split(',') if v.strip()]
        else:
            filter_varieties = DEFAULT_CONFIG['filter_varieties']

        # 转换 max_contracts
        if max_contracts:
            max_contracts = int(max_contracts)

        logger.info(f"收到信号请求 - 品种: {filter_varieties}, 最大合约数: {max_contracts}")

        # 创建数据获取器
        fetcher = MainContractDataFetcher(
            output_dir=DEFAULT_CONFIG['output_dir'],
            filter_varieties=filter_varieties
        )

        # 运行数据获取
        result = run_fetcher_silent(fetcher, delay=delay, max_contracts=max_contracts)

        # 构建响应
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'signals': result['signals'],
            'statistics': {
                'total_contracts': result['total_contracts'],
                'success_count': result['success_count'],
                'fail_count': result['fail_count'],
                'signal_count': len(result['signals'])
            },
            'config': {
                'filter_varieties': filter_varieties,
                'strategy_config': fetcher.strategy_config
            }
        }

        logger.info(f"信号获取成功 - 发现 {len(result['signals'])} 个信号")
        return jsonify(response)

    except Exception as e:
        logger.error(f"获取信号失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/varieties', methods=['GET'])
def get_varieties():
    """获取支持的品种列表"""
    fetcher = MainContractDataFetcher()

    varieties_info = []
    for code, multiplier in fetcher.contract_multipliers.items():
        variety_names = {
            'V': 'PVC',
            'M': '豆粕',
            'C': '玉米',
            'TA': 'PTA',
            'CF': '棉花',
            'RB': '螺纹钢',
            'B': '豆二'
        }
        varieties_info.append({
            'code': code,
            'name': variety_names.get(code, '未知'),
            'multiplier': multiplier
        })

    return jsonify({
        'success': True,
        'varieties': varieties_info,
        'default_filter': DEFAULT_CONFIG['filter_varieties']
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取当前配置"""
    fetcher = MainContractDataFetcher()

    return jsonify({
        'success': True,
        'config': {
            'output_dir': DEFAULT_CONFIG['output_dir'],
            'filter_varieties': DEFAULT_CONFIG['filter_varieties'],
            'strategy_config': fetcher.strategy_config,
            'contract_multipliers': fetcher.contract_multipliers
        }
    })


def run_fetcher_silent(fetcher, delay=0.5, max_contracts=None):
    """
    静默运行数据获取器，不打印输出，只返回结果
    """
    # 获取主力合约列表
    main_contracts = fetcher.get_main_contracts()

    if not main_contracts:
        return {
            'signals': [],
            'total_contracts': 0,
            'success_count': 0,
            'fail_count': 0
        }

    # 过滤品种
    if fetcher.filter_varieties:
        filtered_contracts = []
        for symbol in main_contracts:
            variety_code = fetcher.extract_variety_code(symbol)
            if variety_code in fetcher.filter_varieties:
                filtered_contracts.append(symbol)
        main_contracts = filtered_contracts

    # 限制合约数量
    if max_contracts:
        main_contracts = main_contracts[:max_contracts]

    success_count = 0
    fail_count = 0
    all_signals = []

    for symbol in main_contracts:
        # 获取数据
        df = fetcher.fetch_60min_data(symbol)

        if df is not None and len(df) > 0:
            # 保存数据
            filename = f"{symbol}_60min.csv"
            if fetcher.save_to_csv(df, filename):
                success_count += 1

                # 检查信号
                df_prepared = fetcher.prepare_data_for_signal(df.copy())
                if df_prepared is not None:
                    signals = fetcher.check_signal(df_prepared, symbol)
                    if signals:
                        # 转换 datetime 为字符串
                        for signal in signals:
                            if hasattr(signal['signal_time'], 'isoformat'):
                                signal['signal_time'] = signal['signal_time'].isoformat()
                        all_signals.extend(signals)
            else:
                fail_count += 1
        else:
            fail_count += 1

        # 延迟
        time.sleep(delay)

    return {
        'signals': all_signals,
        'total_contracts': len(main_contracts),
        'success_count': success_count,
        'fail_count': fail_count
    }


if __name__ == '__main__':
    # 启动服务
    port = 5001
    logger.info("=" * 80)
    logger.info(f"期货交易信号API服务启动 - http://localhost:{port}")
    logger.info(f"默认品种: {DEFAULT_CONFIG['filter_varieties']}")
    logger.info(f"数据目录: {DEFAULT_CONFIG['output_dir']}")
    logger.info("=" * 80)
    logger.info("API 端点:")
    logger.info(f"  - GET  /                    服务信息")
    logger.info(f"  - GET  /api/health          健康检查")
    logger.info(f"  - GET  /api/signals         获取交易信号")
    logger.info(f"  - GET  /api/varieties       获取品种列表")
    logger.info(f"  - GET  /api/config          获取配置信息")
    logger.info("=" * 80)
    logger.info("示例请求:")
    logger.info(f"  curl http://localhost:{port}/api/signals?varieties=V,M,C")
    logger.info("=" * 80)

    app.run(host='0.0.0.0', port=port, debug=False)
