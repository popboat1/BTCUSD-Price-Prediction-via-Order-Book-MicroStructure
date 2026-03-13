"use client";

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
    createChart,
    ColorType,
    CrosshairMode,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    LineData,
    UTCTimestamp
} from 'lightweight-charts';
import { Activity, TrendingUp, TrendingDown, Clock, AlignRight, Scaling, BarChart2, LineChart as LineIcon } from 'lucide-react';

interface Tick {
    time: number;
    price: number;
    prediction: number | null;
}

interface OrderBookState {
    bids: string[][];
    asks: string[][];
}

interface PredictionResponse {
    current_price: number;
    predicted_return_bps: number;
    predicted_future_price: number;
    signal: 'BULLISH' | 'BEARISH' | 'WAITING';
}

interface BinanceDepthUpdate {
    lastUpdateId: number;
    bids: string[][];
    asks: string[][];
}

export default function QuantDashboard() {
    const [currentMetrics, setCurrentMetrics] = useState({ price: 0, spread: 0, wobi: 0, signal: 'WAITING' });
    const [isConnected, setIsConnected] = useState(false);
    const [timeframe, setTimeframe] = useState<number>(1);
    const [isLogScale, setIsLogScale] = useState(false);
    const [chartType, setChartType] = useState<'candle' | 'line'>('line');

    const [orderBook, setOrderBook] = useState<OrderBookState>({ bids: [], asks: [] });

    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    const mainSeriesRef = useRef<ISeriesApi<"Candlestick"> | ISeriesApi<"Line"> | null>(null);
    const predictionSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

    const lastApiCallMs = useRef<number>(0);
    const lastOrderBookUpdate = useRef<number>(0);
    const historyRef = useRef<Tick[]>([]);

    useEffect(() => {
        if (chartRef.current) {
            chartRef.current.priceScale('right').applyOptions({ mode: isLogScale ? 2 : 0 });
        }
    }, [isLogScale]);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#0f172a' }, textColor: '#cbd5e1' },
            grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
            crosshair: { mode: CrosshairMode.Normal },
            rightPriceScale: { mode: isLogScale ? 2 : 0, autoScale: true },
            timeScale: { timeVisible: true, secondsVisible: true },
            width: chartContainerRef.current.clientWidth,
            height: 500,
        });

        chartRef.current = chart;

        if (chartType === 'candle') {
            mainSeriesRef.current = chart.addCandlestickSeries({
                upColor: '#10b981', downColor: '#ef4444', borderVisible: false,
                wickUpColor: '#10b981', wickDownColor: '#ef4444',
            });
        } else {
            mainSeriesRef.current = chart.addLineSeries({ color: '#3b82f6', lineWidth: 2 });
        }

        predictionSeriesRef.current = chart.addLineSeries({
            color: '#a855f7', lineWidth: 2, lineStyle: 2, crosshairMarkerVisible: false,
        });

        let currentCandle: CandlestickData | null = null;

        const redrawFromMemory = () => {
            const mainSeries = mainSeriesRef.current;
            const predSeries = predictionSeriesRef.current;
            if (historyRef.current.length === 0 || !mainSeries || !predSeries) return;

            const historicalMain: (CandlestickData | LineData)[] = [];
            const historicalPreds: LineData[] = [];
            let tempCandle: CandlestickData | null = null;

            const sortedHistory = [...historyRef.current].sort((a, b) => a.time - b.time);

            for (const tick of sortedHistory) {
                const candleTime = (Math.floor(tick.time / timeframe) * timeframe) as UTCTimestamp;

                if (!tempCandle || tempCandle.time !== candleTime) {
                    if (tempCandle) {
                        historicalMain.push(chartType === 'candle' ? tempCandle : { time: tempCandle.time, value: tempCandle.close });
                    }
                    tempCandle = { time: candleTime, open: tick.price, high: tick.price, low: tick.price, close: tick.price };
                } else {
                    tempCandle.high = Math.max(tempCandle.high, tick.price);
                    tempCandle.low = Math.min(tempCandle.low, tick.price);
                    tempCandle.close = tick.price;
                }

                if (tick.prediction !== null) {
                    historicalPreds.push({ time: candleTime, value: tick.prediction });
                }
            }

            if (tempCandle) {
                historicalMain.push(chartType === 'candle' ? tempCandle : { time: tempCandle.time, value: tempCandle.close });
                currentCandle = { ...tempCandle };
            }

            if (chartType === 'candle') {
                (mainSeries as ISeriesApi<"Candlestick">).setData(historicalMain as CandlestickData[]);
            } else {
                (mainSeries as ISeriesApi<"Line">).setData(historicalMain as LineData[]);
            }

            const sortedPreds = historicalPreds.sort((a, b) => (a.time as number) - (b.time as number));
            const uniquePreds = Array.from(new Map(sortedPreds.map(item => [item.time, item])).values());
            predSeries.setData(uniquePreds);

            chart.timeScale().fitContent();
        };

        axios.get<string[][]>('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1s&limit=1000')
            .then(res => {
                res.data.forEach(d => historyRef.current.push({ time: Math.floor(Number(d[0]) / 1000), price: parseFloat(d[4]), prediction: null }));
                redrawFromMemory();
            }).catch(err => console.error(err));

        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);

        const ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms');
        ws.onopen = () => setIsConnected(true);
        ws.onclose = () => setIsConnected(false);

        ws.onmessage = async (event: MessageEvent) => {
            const data: BinanceDepthUpdate = JSON.parse(event.data);
            const nowMs = Date.now();

            if (nowMs - lastOrderBookUpdate.current > 500) {
                setOrderBook({ bids: data.bids.slice(0, 10), asks: data.asks.slice(0, 10) });
                lastOrderBookUpdate.current = nowMs;
            }

            const midPrice = (parseFloat(data.asks[0][0]) + parseFloat(data.bids[0][0])) / 2;
            const nowSeconds = Math.floor(nowMs / 1000);
            const candleTime = (Math.floor(nowSeconds / timeframe) * timeframe) as UTCTimestamp;

            if (mainSeriesRef.current) {
                if (!currentCandle || currentCandle.time !== candleTime) {
                    currentCandle = { time: candleTime, open: midPrice, high: midPrice, low: midPrice, close: midPrice };
                } else {
                    currentCandle.high = Math.max(currentCandle.high, midPrice);
                    currentCandle.low = Math.min(currentCandle.low, midPrice);
                    currentCandle.close = midPrice;
                }

                if (chartType === 'candle') {
                    (mainSeriesRef.current as ISeriesApi<"Candlestick">).update(currentCandle);
                } else {
                    (mainSeriesRef.current as ISeriesApi<"Line">).update({ time: candleTime, value: midPrice });
                }
            }

            historyRef.current.push({ time: nowSeconds, price: midPrice, prediction: null });
            if (historyRef.current.length > 5000) historyRef.current.shift();

            if (nowMs - lastApiCallMs.current < 1000) return;
            lastApiCallMs.current = nowMs;

            let wb = 0; let wa = 0; let tv = 0;
            for (let i = 0; i < 10; i++) {
                const b = parseFloat(data.bids[i][1]); const a = parseFloat(data.asks[i][1]);
                tv += (b + a); const w = 1.0 - (i * 0.1); wb += (b * w); wa += (a * w);
            }
            const wobi = (wb + wa) > 0 ? (wb - wa) / (wb + wa) : 0;

            try {
                const response = await axios.post<PredictionResponse>('/api/predict', {
                    current_price: midPrice, spread: parseFloat(data.asks[0][0]) - parseFloat(data.bids[0][0]), total_vol: tv, wobi
                });

                if (predictionSeriesRef.current) {
                    predictionSeriesRef.current.update({ time: candleTime, value: response.data.predicted_future_price });
                }
                historyRef.current[historyRef.current.length - 1].prediction = response.data.predicted_future_price;
                setCurrentMetrics({
                    price: midPrice,
                    spread: parseFloat(data.asks[0][0]) - parseFloat(data.bids[0][0]),
                    wobi,
                    signal: response.data.signal
                });
            } catch (e) { console.error(e); }
        };

        return () => {
            window.removeEventListener('resize', handleResize);
            ws.close();
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
        };
    }, [timeframe, isLogScale, chartType]);

    const maxTotalVol = Math.max(...orderBook.bids.map(b => parseFloat(b[1])), ...orderBook.asks.map(a => parseFloat(a[1])), 0.0001);
    return (
        <div className="flex flex-col min-h-screen p-4 font-sans bg-slate-950 text-slate-50 md:p-8">
            <header className="flex flex-col items-start justify-between gap-4 mb-8 md:flex-row md:items-center">
                <div>
                    <h1 className="text-2xl font-bold bg-blue-400 bg-clip-text text-transparent md:text-3xl">
                        Linear Alpha
                    </h1>
                    <div className="flex items-center gap-2 mt-1">
                        <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500 animate-pulse'}`}></span>
                        <p className="text-sm text-slate-400">
                            {isConnected ? 'Live' : 'Connecting...'} • Linear Regression Model • BTC/USDT Microstructure
                        </p>
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                    <div className="flex items-center gap-1 p-1 border rounded-lg bg-slate-900 border-slate-800">
                        <button onClick={() => setChartType('candle')} className={`cursor-pointer p-1.5 rounded-md transition-all ${chartType === 'candle' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`} title="Candlestick">
                            <BarChart2 size={18} />
                        </button>
                        <button onClick={() => setChartType('line')} className={`cursor-pointer p-1.5 rounded-md transition-all ${chartType === 'line' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'}`} title="Line">
                            <LineIcon size={18} />
                        </button>
                    </div>

                    <button onClick={() => setIsLogScale(!isLogScale)} className={`cursor-pointer flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm transition-colors ${isLogScale ? 'bg-blue-600/20 border-blue-500 text-blue-400' : 'bg-slate-900 border-slate-700 text-slate-400'}`}>
                        <Scaling size={16} /> {isLogScale ? 'Log' : 'Auto'}
                    </button>

                    <div className="flex items-center gap-1 p-1 border rounded-lg bg-slate-900 border-slate-800">
                        <Clock size={14} className="ml-1 text-slate-400" />
                        {[1, 5, 15, 60].map((tf) => (
                            <button key={tf} onClick={() => setTimeframe(tf)} className={`px-3 py-1 text-xs rounded-md cursor-pointer ${timeframe === tf ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`}>
                                {tf === 60 ? '1m' : `${tf}s`}
                            </button>
                        ))}
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-2 gap-6 mb-6 lg:grid-cols-4">
                {[
                    { label: 'Current Price', value: `$${currentMetrics.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, icon: <Activity size={14} />, color: 'text-slate-50' },
                    { label: 'Live Spread', value: `$${currentMetrics.spread.toFixed(2)}`, icon: null, color: 'text-slate-50' },
                    { label: 'WOBI Score', value: currentMetrics.wobi.toFixed(4), icon: null, color: currentMetrics.wobi > 0 ? 'text-emerald-400' : 'text-red-400' },
                    { label: 'Model Signal', value: currentMetrics.signal, icon: currentMetrics.signal === 'BULLISH' ? <TrendingUp size={16} /> : <TrendingDown size={16} />, color: currentMetrics.signal === 'BULLISH' ? 'text-emerald-400' : 'text-red-400' }
                ].map((metric, idx) => (
                    <div key={idx} className="p-4 shadow-lg bg-slate-900 border border-slate-800 rounded-xl">
                        <div className="flex items-center gap-2 mb-1 text-xs text-slate-400">{metric.icon} {metric.label}</div>
                        <div className={`text-lg md:text-xl font-mono font-bold ${metric.color}`}>{metric.value}</div>
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 flex-1 min-h-150">
                <div className="flex flex-col p-4 shadow-xl bg-slate-900 border border-slate-800 rounded-xl relative lg:col-span-3">
                    <h2 className="text-sm font-semibold mb-4">{timeframe === 60 ? '1-Min' : `${timeframe}-Sec`} Horizon</h2>
                    <div ref={chartContainerRef} className="flex-1 w-full" />
                </div>

                <aside className="flex flex-col p-4 shadow-xl bg-slate-900 border border-slate-800 rounded-xl lg:col-span-1">
                    <h2 className="flex items-center gap-2 pb-2 mb-4 text-sm font-semibold border-b text-slate-300 border-slate-800">
                        <AlignRight size={16} /> Market Depth
                    </h2>
                    <div className="flex flex-col flex-1 gap-1 overflow-hidden font-mono text-xs">
                        <div className="flex flex-col-reverse gap-0.5">
                            {orderBook.asks.map((ask, i) => (
                                <div key={`ask-${i}`} className="relative flex justify-between px-2 py-0.5 group">
                                    <div className="absolute right-0 top-0 bottom-0 bg-red-500/10" style={{ width: `${(parseFloat(ask[1]) / maxTotalVol) * 100}%` }} />
                                    <span className="relative z-10 text-red-400">{parseFloat(ask[0]).toFixed(2)}</span>
                                    <span className="relative z-10 text-slate-300">{parseFloat(ask[1]).toFixed(4)}</span>
                                </div>
                            ))}
                        </div>
                        <div className="py-3 my-1 font-bold text-center border-y border-slate-800 bg-slate-950/40">
                            <div className="text-lg text-slate-100">${currentMetrics.price.toFixed(2)}</div>
                        </div>
                        <div className="flex flex-col gap-0.5">
                            {orderBook.bids.map((bid, i) => (
                                <div key={`bid-${i}`} className="relative flex justify-between px-2 py-0.5 group">
                                    <div className="absolute right-0 top-0 bottom-0 bg-emerald-500/10" style={{ width: `${(parseFloat(bid[1]) / maxTotalVol) * 100}%` }} />
                                    <span className="relative z-10 text-emerald-400">{parseFloat(bid[0]).toFixed(2)}</span>
                                    <span className="relative z-10 text-slate-300">{parseFloat(bid[1]).toFixed(4)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>
            </div>
        </div>
    );
}