# EA_SCALPER_XAUUSD - Plano de Implementações Futuras
## Guia Completo para Agentes e Desenvolvedores

---

## 📋 Visão Geral

Este documento contém o plano detalhado de implementações futuras para o EA_SCALPER_XAUUSD. 
**Objetivo**: Servir como referência para qualquer agente ou desenvolvedor que trabalhe no projeto.

**Data**: 2024-11-29
**Versão**: 1.0

---

## 🎯 Prioridades de Implementação

| Prioridade | Módulo | Impacto | Esforço | Backtestável |
|------------|--------|---------|---------|--------------|
| **1** | Volume Profile (POC/VAH/VAL) | 🔥 ALTO | Médio | ✅ 100% |
| **2** | R-Multiple Tracker | 🔥 ALTO | Baixo | ✅ 100% |
| **3** | Risk of Ruin Calculator | 🔥 ALTO | Baixo | ✅ 100% |
| **4** | Volume Delta (Tick Rule) | Médio | Médio | ⚠️ 90% |
| **5** | Imbalance Detection | Médio | Médio | ⚠️ 85% |

---

## 1. 📊 Volume Profile (POC/VAH/VAL)

### 1.1 O Que É

Volume Profile mostra **onde** o volume foi transacionado (por preço), não **quando** (por tempo).

```
Distribuição de Volume por Preço:

Preço   |  Volume  |  Visualização
--------|----------|------------------
$2050   |   500    |  ████
$2048   |   800    |  ██████
$2046   |  1500    |  ███████████████  ← POC (Point of Control)
$2044   |  1200    |  ████████████
$2042   |   900    |  ███████
$2040   |   400    |  ████

VAH (Value Area High) = $2048
POC (Point of Control) = $2046
VAL (Value Area Low)   = $2042
Value Area = 70% do volume total
```

### 1.2 Por Que É Importante

1. **POC como Suporte/Resistência**: Preço tende a gravitar em torno do POC
2. **Value Area como Range**: 70% do tempo o preço fica dentro da VA
3. **Confluência com SMC**: OB dentro de Value Area = mais forte
4. **Breakout Confirmation**: Preço fora da VA com volume = movimento real

### 1.3 Implementação Python

**Localização**: `Python_Agent_Hub/ml_pipeline/indicators/volume_profile.py`

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class VolumeProfileResult:
    poc: float                    # Point of Control
    vah: float                    # Value Area High
    val: float                    # Value Area Low
    profile: Dict[float, float]   # Price -> Volume mapping
    value_area_pct: float = 0.70  # 70% default

class VolumeProfileCalculator:
    """
    Calcula Volume Profile com POC, VAH, VAL.
    100% backtestável - usa apenas OHLCV.
    """
    
    def __init__(self, price_bins: int = 50, value_area_pct: float = 0.70):
        self.price_bins = price_bins
        self.value_area_pct = value_area_pct
    
    def calculate(self, df: pd.DataFrame, lookback: int = 200) -> VolumeProfileResult:
        """
        Calcula Volume Profile para os últimos N bars.
        
        Args:
            df: DataFrame com 'high', 'low', 'close', 'volume'
            lookback: Número de barras para análise
        
        Returns:
            VolumeProfileResult com POC, VAH, VAL
        """
        data = df.tail(lookback)
        
        # Determinar range de preço
        price_min = data['low'].min()
        price_max = data['high'].max()
        bin_size = (price_max - price_min) / self.price_bins
        
        if bin_size == 0:
            return VolumeProfileResult(
                poc=data['close'].iloc[-1],
                vah=price_max,
                val=price_min,
                profile={}
            )
        
        # Distribuir volume por preço
        volume_at_price = {}
        
        for idx in range(len(data)):
            row = data.iloc[idx]
            bar_high = row['high']
            bar_low = row['low']
            bar_vol = row['volume']
            bar_range = bar_high - bar_low
            
            if bar_range == 0:
                # Doji - todo volume no close
                bin_key = self._price_to_bin(row['close'], price_min, bin_size)
                volume_at_price[bin_key] = volume_at_price.get(bin_key, 0) + bar_vol
                continue
            
            # Distribuir volume proporcionalmente
            # TPO approach: dividir igualmente nos bins que a barra tocou
            bins_touched = []
            price = bar_low
            while price <= bar_high:
                bin_key = self._price_to_bin(price, price_min, bin_size)
                if bin_key not in bins_touched:
                    bins_touched.append(bin_key)
                price += bin_size
            
            vol_per_bin = bar_vol / len(bins_touched) if bins_touched else 0
            for bin_key in bins_touched:
                volume_at_price[bin_key] = volume_at_price.get(bin_key, 0) + vol_per_bin
        
        if not volume_at_price:
            return VolumeProfileResult(
                poc=data['close'].iloc[-1],
                vah=price_max,
                val=price_min,
                profile={}
            )
        
        # POC = preço com maior volume
        poc_bin = max(volume_at_price, key=volume_at_price.get)
        poc = price_min + (poc_bin + 0.5) * bin_size
        
        # Value Area (70% do volume, expandindo do POC)
        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * self.value_area_pct
        
        # Expandir do POC para cima e para baixo
        va_bins = {poc_bin}
        current_volume = volume_at_price.get(poc_bin, 0)
        
        bins_sorted = sorted(volume_at_price.keys())
        poc_idx = bins_sorted.index(poc_bin) if poc_bin in bins_sorted else 0
        
        up_idx = poc_idx + 1
        down_idx = poc_idx - 1
        
        while current_volume < target_volume:
            up_vol = volume_at_price.get(bins_sorted[up_idx], 0) if up_idx < len(bins_sorted) else 0
            down_vol = volume_at_price.get(bins_sorted[down_idx], 0) if down_idx >= 0 else 0
            
            if up_vol == 0 and down_vol == 0:
                break
            
            if up_vol >= down_vol and up_idx < len(bins_sorted):
                va_bins.add(bins_sorted[up_idx])
                current_volume += up_vol
                up_idx += 1
            elif down_idx >= 0:
                va_bins.add(bins_sorted[down_idx])
                current_volume += down_vol
                down_idx -= 1
            else:
                break
        
        # VAH e VAL
        vah_bin = max(va_bins)
        val_bin = min(va_bins)
        vah = price_min + (vah_bin + 1) * bin_size
        val = price_min + val_bin * bin_size
        
        return VolumeProfileResult(
            poc=poc,
            vah=vah,
            val=val,
            profile={price_min + (k + 0.5) * bin_size: v for k, v in volume_at_price.items()},
            value_area_pct=self.value_area_pct
        )
    
    def _price_to_bin(self, price: float, price_min: float, bin_size: float) -> int:
        return int((price - price_min) / bin_size)
    
    def get_features(self, df: pd.DataFrame, current_price: float, lookback: int = 200) -> Dict[str, float]:
        """
        Retorna features para ML baseadas no Volume Profile.
        """
        result = self.calculate(df, lookback)
        atr = self._calculate_atr(df, 14)
        
        return {
            'vp_poc_distance': (current_price - result.poc) / atr if atr > 0 else 0,
            'vp_vah_distance': (current_price - result.vah) / atr if atr > 0 else 0,
            'vp_val_distance': (current_price - result.val) / atr if atr > 0 else 0,
            'vp_in_value_area': 1.0 if result.val <= current_price <= result.vah else 0.0,
            'vp_above_poc': 1.0 if current_price > result.poc else 0.0,
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
```

### 1.4 Uso Como Feature para ML

```python
# Em feature_engineering.py, adicionar:
from indicators.volume_profile import VolumeProfileCalculator

vp_calc = VolumeProfileCalculator(price_bins=50)
vp_features = vp_calc.get_features(df, current_price=df['close'].iloc[-1])

# Features resultantes:
# - vp_poc_distance: Distância ao POC em ATRs (-2 a +2 típico)
# - vp_vah_distance: Distância ao VAH em ATRs
# - vp_val_distance: Distância ao VAL em ATRs
# - vp_in_value_area: 1.0 se preço dentro da VA, 0.0 se fora
# - vp_above_poc: 1.0 se acima do POC, 0.0 se abaixo
```

### 1.5 Integração com SMC Existente

```
CONFLUÊNCIA IDEAL:

1. Preço retorna ao POC + OB presente = Trade de alta qualidade
2. Sweep de liquidez + retorno à Value Area = Confirmação extra
3. FVG dentro da Value Area = FVG mais significativo

REGRAS:
- Se preço toca POC pela primeira vez = Forte S/R
- Se preço rompe VAH/VAL com volume = Breakout válido
- Preço muito fora da VA = Mean reversion esperada
```

---

## 2. 📈 R-Multiple Tracker (Van Tharp)

### 2.1 O Que É

R-Multiple mede cada trade em **unidades de risco inicial**.

```
R = Risco Inicial (em $)

Trade 1: Risco $100, Ganho $250 = +2.5R
Trade 2: Risco $100, Perda $80  = -0.8R
Trade 3: Risco $100, Ganho $150 = +1.5R
```

### 2.2 Por Que É Importante

1. **Normalização**: Compara trades de diferentes tamanhos
2. **Expectancy**: Média de R = expectativa por trade
3. **SQN**: System Quality Number = qualidade do sistema
4. **Position Sizing**: Baseado em múltiplos de R

### 2.3 Métricas Derivadas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Expectancy** | Média(R) | Lucro esperado por trade em R |
| **SQN** | (Média(R) × √N) / Std(R) | Qualidade do sistema |
| **Profit Factor** | Soma(R+) / \|Soma(R-)\| | Lucro vs Perda |

**Classificação SQN (Van Tharp)**:
| SQN | Classificação |
|-----|--------------|
| < 1.6 | Poor |
| 1.6 - 2.0 | Average |
| 2.0 - 3.0 | Good |
| 3.0 - 5.0 | Excellent |
| 5.0 - 7.0 | Superb |
| > 7.0 | Holy Grail |

### 2.4 Implementação Python

**Localização**: `Python_Agent_Hub/ml_pipeline/risk/r_multiple_tracker.py`

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    size: float
    
    @property
    def initial_risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def pnl(self) -> float:
        if self.direction == 'long':
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price
    
    @property
    def r_multiple(self) -> float:
        if self.initial_risk == 0:
            return 0
        return self.pnl / self.initial_risk

@dataclass
class RMultipleStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    expectancy: float          # Média de R
    sqn: float                 # System Quality Number
    profit_factor: float
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    max_r: float
    min_r: float
    std_r: float
    all_r_multiples: List[float]
    sqn_rating: str

class RMultipleTracker:
    """
    Rastreador de R-Múltiplos baseado na metodologia Van Tharp.
    Essencial para medir a qualidade do sistema de trading.
    """
    
    def __init__(self):
        self.trades: List[Trade] = []
    
    def record_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        size: float = 1.0
    ) -> float:
        """Registra um trade e retorna o R-múltiplo."""
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            size=size
        )
        self.trades.append(trade)
        return trade.r_multiple
    
    def get_stats(self) -> RMultipleStats:
        """Calcula estatísticas completas de R-múltiplos."""
        if not self.trades:
            return RMultipleStats(
                total_trades=0, winning_trades=0, losing_trades=0,
                expectancy=0, sqn=0, profit_factor=0, win_rate=0,
                avg_win_r=0, avg_loss_r=0, max_r=0, min_r=0, std_r=0,
                all_r_multiples=[], sqn_rating='No Data'
            )
        
        r_multiples = [t.r_multiple for t in self.trades]
        winners = [r for r in r_multiples if r > 0]
        losers = [r for r in r_multiples if r < 0]
        
        n = len(r_multiples)
        expectancy = np.mean(r_multiples)
        std_r = np.std(r_multiples) if n > 1 else 0
        
        # SQN = (Expectancy * sqrt(N)) / StdDev
        sqn = (expectancy * np.sqrt(n)) / std_r if std_r > 0 else 0
        
        # Profit Factor
        sum_winners = sum(winners) if winners else 0
        sum_losers = abs(sum(losers)) if losers else 0
        profit_factor = sum_winners / sum_losers if sum_losers > 0 else float('inf')
        
        # Rating
        sqn_rating = self._get_sqn_rating(sqn)
        
        return RMultipleStats(
            total_trades=n,
            winning_trades=len(winners),
            losing_trades=len(losers),
            expectancy=expectancy,
            sqn=sqn,
            profit_factor=profit_factor,
            win_rate=len(winners) / n if n > 0 else 0,
            avg_win_r=np.mean(winners) if winners else 0,
            avg_loss_r=np.mean(losers) if losers else 0,
            max_r=max(r_multiples),
            min_r=min(r_multiples),
            std_r=std_r,
            all_r_multiples=r_multiples,
            sqn_rating=sqn_rating
        )
    
    def _get_sqn_rating(self, sqn: float) -> str:
        if sqn < 1.6:
            return 'Poor'
        elif sqn < 2.0:
            return 'Average'
        elif sqn < 3.0:
            return 'Good'
        elif sqn < 5.0:
            return 'Excellent'
        elif sqn < 7.0:
            return 'Superb'
        else:
            return 'Holy Grail'
    
    def get_rolling_expectancy(self, window: int = 50) -> List[float]:
        """Calcula expectancy rolling para detectar degradação."""
        r_multiples = [t.r_multiple for t in self.trades]
        if len(r_multiples) < window:
            return [np.mean(r_multiples)] if r_multiples else [0]
        
        rolling = []
        for i in range(window, len(r_multiples) + 1):
            rolling.append(np.mean(r_multiples[i-window:i]))
        return rolling
    
    def should_reduce_risk(self, min_sqn: float = 1.6, window: int = 30) -> bool:
        """Recomenda reduzir risco se SQN cair abaixo do mínimo."""
        if len(self.trades) < window:
            return False
        
        recent_trades = self.trades[-window:]
        r_multiples = [t.r_multiple for t in recent_trades]
        expectancy = np.mean(r_multiples)
        std_r = np.std(r_multiples)
        sqn = (expectancy * np.sqrt(len(r_multiples))) / std_r if std_r > 0 else 0
        
        return sqn < min_sqn
    
    def to_dataframe(self) -> pd.DataFrame:
        """Exporta trades como DataFrame."""
        return pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'initial_risk': t.initial_risk,
                'pnl': t.pnl,
                'r_multiple': t.r_multiple
            }
            for t in self.trades
        ])
    
    def print_report(self):
        """Imprime relatório formatado."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("R-MULTIPLE ANALYSIS REPORT (Van Tharp Method)")
        print("="*60)
        
        print(f"\n--- Performance ---")
        print(f"Total Trades:    {stats.total_trades}")
        print(f"Winning:         {stats.winning_trades} ({stats.win_rate*100:.1f}%)")
        print(f"Losing:          {stats.losing_trades}")
        
        print(f"\n--- R-Multiple Metrics ---")
        print(f"Expectancy:      {stats.expectancy:.3f}R per trade")
        print(f"SQN:             {stats.sqn:.2f} ({stats.sqn_rating})")
        print(f"Profit Factor:   {stats.profit_factor:.2f}")
        
        print(f"\n--- Distribution ---")
        print(f"Avg Win:         +{stats.avg_win_r:.2f}R")
        print(f"Avg Loss:        {stats.avg_loss_r:.2f}R")
        print(f"Best Trade:      +{stats.max_r:.2f}R")
        print(f"Worst Trade:     {stats.min_r:.2f}R")
        print(f"Std Dev:         {stats.std_r:.2f}R")
        
        print("="*60)
```

### 2.5 Integração com FTMO Simulator

```python
# Em ftmo_simulator.py, adicionar:
from risk.r_multiple_tracker import RMultipleTracker

class FTMOSimulator:
    def __init__(self):
        self.r_tracker = RMultipleTracker()
    
    def record_trade(self, trade: TradeResult):
        # Registrar no R-Tracker
        self.r_tracker.record_trade(
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,  # Precisa adicionar ao TradeResult
            size=trade.size
        )
        # ... resto do código
```

---

## 3. ⚠️ Risk of Ruin Calculator (Ralph Vince)

### 3.1 O Que É

Probabilidade de perder X% da conta antes de atingir lucro alvo.

```
EXEMPLO:
- Win Rate: 60%
- Avg Win: 2R
- Avg Loss: 1R
- Risk per Trade: 1%
- Ruin Level: 20% drawdown

PERGUNTA: Qual a chance de perder 20% antes de lucrar 50%?
RESPOSTA: Monte Carlo dirá (ex: 2.3% de ruin)
```

### 3.2 Por Que É Importante

1. **Validação Antes de Live**: Se RoR > 5%, sistema é perigoso
2. **Position Sizing**: Ajustar risco para RoR aceitável
3. **Confiança**: Saber que chance de quebrar é < 1%

### 3.3 Implementação Python

**Localização**: `Python_Agent_Hub/ml_pipeline/risk/risk_of_ruin.py`

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import scipy.stats as stats

@dataclass
class RiskOfRuinResult:
    risk_of_ruin: float           # Probabilidade de ruin (0-1)
    probability_of_success: float  # Probabilidade de atingir target
    inconclusive: float           # Não atingiu nem ruin nem target
    median_max_drawdown: float    # DD mediano nas simulações
    percentile_95_drawdown: float # DD no percentil 95
    optimal_kelly_fraction: float # Kelly Criterion
    half_kelly: float             # Metade do Kelly (mais conservador)
    simulations: int
    trades_per_sim: int

class RiskOfRuinCalculator:
    """
    Calculadora de Risk of Ruin usando Monte Carlo.
    Baseado em Ralph Vince e Van Tharp.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def calculate_monte_carlo(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        risk_per_trade: float,
        ruin_level: float = 0.20,      # 20% DD = ruin
        target_level: float = 0.50,     # 50% profit = success
        trades_per_sim: int = 200,
        simulations: int = 10000,
        use_distribution: bool = True   # Usar distribuição realista
    ) -> RiskOfRuinResult:
        """
        Calcula Risk of Ruin via Monte Carlo.
        
        Args:
            win_rate: Taxa de acerto (0-1)
            avg_win_r: Ganho médio em R-múltiplos
            avg_loss_r: Perda média em R-múltiplos (valor absoluto)
            risk_per_trade: Fração da conta arriscada (0.01 = 1%)
            ruin_level: Drawdown que define ruin (0.20 = 20%)
            target_level: Lucro alvo (0.50 = 50%)
            trades_per_sim: Trades por simulação
            simulations: Número de simulações Monte Carlo
            use_distribution: Se True, usa distribuição normal para R
        
        Returns:
            RiskOfRuinResult com todas as métricas
        """
        ruined = 0
        succeeded = 0
        max_drawdowns = []
        
        for _ in range(simulations):
            equity = 1.0
            peak_equity = 1.0
            max_dd_this_sim = 0.0
            
            for _ in range(trades_per_sim):
                # Gerar resultado do trade
                if np.random.random() < win_rate:
                    if use_distribution:
                        # R com variação (mais realista)
                        r_result = np.random.normal(avg_win_r, avg_win_r * 0.3)
                        r_result = max(0.1, r_result)  # Mínimo 0.1R
                    else:
                        r_result = avg_win_r
                    equity += risk_per_trade * r_result
                else:
                    if use_distribution:
                        r_result = np.random.normal(avg_loss_r, avg_loss_r * 0.3)
                        r_result = max(0.1, r_result)
                    else:
                        r_result = avg_loss_r
                    equity -= risk_per_trade * r_result
                
                # Atualizar peak e drawdown
                if equity > peak_equity:
                    peak_equity = equity
                
                drawdown = (peak_equity - equity) / peak_equity
                max_dd_this_sim = max(max_dd_this_sim, drawdown)
                
                # Verificar ruin
                if drawdown >= ruin_level:
                    ruined += 1
                    break
                
                # Verificar sucesso
                profit = (equity - 1.0)
                if profit >= target_level:
                    succeeded += 1
                    break
            
            max_drawdowns.append(max_dd_this_sim)
        
        # Calcular Kelly Criterion
        kelly = self._calculate_kelly(win_rate, avg_win_r, avg_loss_r)
        
        return RiskOfRuinResult(
            risk_of_ruin=ruined / simulations,
            probability_of_success=succeeded / simulations,
            inconclusive=(simulations - ruined - succeeded) / simulations,
            median_max_drawdown=np.median(max_drawdowns),
            percentile_95_drawdown=np.percentile(max_drawdowns, 95),
            optimal_kelly_fraction=kelly,
            half_kelly=kelly / 2,
            simulations=simulations,
            trades_per_sim=trades_per_sim
        )
    
    def _calculate_kelly(
        self, 
        win_rate: float, 
        avg_win_r: float, 
        avg_loss_r: float
    ) -> float:
        """
        Calcula Kelly Criterion.
        f* = (p * b - q) / b
        onde p = win rate, q = 1-p, b = win/loss ratio
        """
        if avg_loss_r == 0:
            return 0
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win_r / avg_loss_r
        
        kelly = (p * b - q) / b
        return max(0, kelly)
    
    def calculate_analytical(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        risk_per_trade: float,
        ruin_level: float = 0.20
    ) -> float:
        """
        Calcula Risk of Ruin usando fórmula analítica (aproximação).
        Menos preciso que Monte Carlo, mas instantâneo.
        """
        # Edge (vantagem)
        edge = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r
        
        if edge <= 0:
            return 1.0  # Sistema perdedor = 100% de ruin
        
        # Unidades de risco até ruin
        units_to_ruin = ruin_level / risk_per_trade
        
        # Fórmula de Gambler's Ruin adaptada
        a = (1 - edge) / (1 + edge) if edge != 0 else 1
        
        if abs(a) >= 1:
            return 1.0
        
        ror = a ** units_to_ruin
        return min(1.0, max(0.0, ror))
    
    def find_safe_risk(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        max_ror: float = 0.01,  # Máximo 1% de ruin
        ruin_level: float = 0.20
    ) -> float:
        """
        Encontra o risco máximo por trade para RoR <= max_ror.
        Usa busca binária.
        """
        low = 0.001   # 0.1%
        high = 0.10   # 10%
        
        for _ in range(20):  # 20 iterações = precisão de ~0.01%
            mid = (low + high) / 2
            result = self.calculate_monte_carlo(
                win_rate=win_rate,
                avg_win_r=avg_win_r,
                avg_loss_r=avg_loss_r,
                risk_per_trade=mid,
                ruin_level=ruin_level,
                simulations=1000  # Menos simulações para speed
            )
            
            if result.risk_of_ruin > max_ror:
                high = mid
            else:
                low = mid
        
        return low
    
    def print_report(self, result: RiskOfRuinResult, params: dict):
        """Imprime relatório formatado."""
        print("\n" + "="*60)
        print("RISK OF RUIN ANALYSIS (Ralph Vince Method)")
        print("="*60)
        
        print(f"\n--- Input Parameters ---")
        print(f"Win Rate:        {params.get('win_rate', 0)*100:.1f}%")
        print(f"Avg Win:         +{params.get('avg_win_r', 0):.2f}R")
        print(f"Avg Loss:        -{params.get('avg_loss_r', 0):.2f}R")
        print(f"Risk per Trade:  {params.get('risk_per_trade', 0)*100:.2f}%")
        print(f"Ruin Level:      {params.get('ruin_level', 0.2)*100:.0f}% drawdown")
        print(f"Target:          {params.get('target_level', 0.5)*100:.0f}% profit")
        
        print(f"\n--- Monte Carlo Results ({result.simulations:,} simulations) ---")
        print(f"Risk of Ruin:    {result.risk_of_ruin*100:.2f}%")
        print(f"P(Success):      {result.probability_of_success*100:.2f}%")
        print(f"Inconclusive:    {result.inconclusive*100:.2f}%")
        
        print(f"\n--- Drawdown Analysis ---")
        print(f"Median Max DD:   {result.median_max_drawdown*100:.1f}%")
        print(f"95th Pctl DD:    {result.percentile_95_drawdown*100:.1f}%")
        
        print(f"\n--- Position Sizing Recommendations ---")
        print(f"Kelly Criterion: {result.optimal_kelly_fraction*100:.2f}%")
        print(f"Half Kelly:      {result.half_kelly*100:.2f}% (recommended)")
        
        # Interpretação
        print(f"\n--- Interpretation ---")
        if result.risk_of_ruin < 0.01:
            print("✅ EXCELLENT: Risk of ruin < 1%. System is very safe.")
        elif result.risk_of_ruin < 0.05:
            print("✅ GOOD: Risk of ruin 1-5%. Acceptable for trading.")
        elif result.risk_of_ruin < 0.10:
            print("⚠️ WARNING: Risk of ruin 5-10%. Consider reducing risk.")
        else:
            print("❌ DANGEROUS: Risk of ruin > 10%. DO NOT TRADE this system!")
        
        print("="*60)
```

### 3.4 Uso Prático

```python
# Antes de iniciar trading:
from risk.risk_of_ruin import RiskOfRuinCalculator

calc = RiskOfRuinCalculator()

# Com métricas do backtest
result = calc.calculate_monte_carlo(
    win_rate=0.65,
    avg_win_r=2.0,
    avg_loss_r=1.0,
    risk_per_trade=0.01,  # 1%
    ruin_level=0.20,      # 20% = FTMO limit
    target_level=0.10,    # 10% = FTMO Phase 1
    simulations=10000
)

calc.print_report(result, {
    'win_rate': 0.65,
    'avg_win_r': 2.0,
    'avg_loss_r': 1.0,
    'risk_per_trade': 0.01,
    'ruin_level': 0.20,
    'target_level': 0.10
})

# Encontrar risco seguro
safe_risk = calc.find_safe_risk(
    win_rate=0.65,
    avg_win_r=2.0,
    avg_loss_r=1.0,
    max_ror=0.01  # Máximo 1% de ruin
)
print(f"Safe risk per trade: {safe_risk*100:.2f}%")
```

---

## 4. 📉 Volume Delta (Tick Rule)

### 4.1 O Que É

Delta = Buy Volume - Sell Volume

**Tick Rule (Lee & Ready 1991)**:
- Se tick sobe → classificar como BUY
- Se tick desce → classificar como SELL
- Mesmo preço → usar última classificação

### 4.2 Limitações

- **Não é 100% preciso**: Tick rule é aproximação
- **Requer tick data**: Nem sempre disponível historicamente
- **Mais útil em tempo real**: Para confirmação

### 4.3 Implementação Python

**Localização**: `Python_Agent_Hub/ml_pipeline/indicators/volume_delta.py`

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DeltaResult:
    delta: float          # Buy - Sell
    buy_volume: float
    sell_volume: float
    delta_pct: float      # Delta / Total Volume

class VolumeDeltaCalculator:
    """
    Calcula Volume Delta usando Tick Rule.
    Aproximação academicamente válida (Lee & Ready 1991).
    
    IMPORTANTE: Requer dados de tick (CopyTicks no MQL5).
    Para backtest histórico, usa aproximação por candle.
    """
    
    def calculate_from_ticks(self, ticks_df: pd.DataFrame) -> DeltaResult:
        """
        Calcula delta de tick data real.
        
        Args:
            ticks_df: DataFrame com 'price' e opcionalmente 'volume'
        """
        if len(ticks_df) < 2:
            return DeltaResult(0, 0, 0, 0)
        
        prices = ticks_df['price'].values
        volumes = ticks_df['volume'].values if 'volume' in ticks_df else np.ones(len(ticks_df))
        
        buy_vol = 0.0
        sell_vol = 0.0
        last_direction = 0  # 1 = buy, -1 = sell
        
        for i in range(1, len(prices)):
            price_change = prices[i] - prices[i-1]
            vol = volumes[i]
            
            if price_change > 0:
                buy_vol += vol
                last_direction = 1
            elif price_change < 0:
                sell_vol += vol
                last_direction = -1
            else:
                # Mesmo preço - usar última direção
                if last_direction == 1:
                    buy_vol += vol
                else:
                    sell_vol += vol
        
        total = buy_vol + sell_vol
        delta = buy_vol - sell_vol
        delta_pct = delta / total if total > 0 else 0
        
        return DeltaResult(
            delta=delta,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            delta_pct=delta_pct
        )
    
    def calculate_from_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aproximação de delta usando OHLCV (para backtest).
        
        Lógica:
        - Candle de alta (close > open): maior parte do volume é buy
        - Candle de baixa (close < open): maior parte do volume é sell
        - Proporção baseada na posição do close no range
        """
        result = df.copy()
        
        # Posição do close no range (0 = low, 1 = high)
        range_size = result['high'] - result['low']
        close_position = np.where(
            range_size > 0,
            (result['close'] - result['low']) / range_size,
            0.5
        )
        
        # Estimar buy/sell volume
        result['buy_volume_est'] = result['volume'] * close_position
        result['sell_volume_est'] = result['volume'] * (1 - close_position)
        result['delta_est'] = result['buy_volume_est'] - result['sell_volume_est']
        result['delta_pct'] = result['delta_est'] / result['volume']
        
        # Delta cumulativo
        result['cumulative_delta'] = result['delta_est'].cumsum()
        
        # Delta rate of change
        result['delta_roc'] = result['cumulative_delta'].diff(5)
        
        return result
    
    def get_features(self, df: pd.DataFrame) -> dict:
        """
        Retorna features para ML baseadas em delta.
        """
        df_delta = self.calculate_from_ohlcv(df)
        
        # Últimos valores
        current_delta_pct = df_delta['delta_pct'].iloc[-1]
        cum_delta = df_delta['cumulative_delta'].iloc[-1]
        delta_roc = df_delta['delta_roc'].iloc[-1]
        
        # Média rolling
        delta_ma = df_delta['delta_pct'].rolling(20).mean().iloc[-1]
        
        # Divergência: preço subindo + delta negativo = bearish divergence
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        delta_change = df_delta['cumulative_delta'].iloc[-1] - df_delta['cumulative_delta'].iloc[-20]
        
        divergence = 0
        if price_change > 0.01 and delta_change < 0:
            divergence = -1  # Bearish divergence
        elif price_change < -0.01 and delta_change > 0:
            divergence = 1   # Bullish divergence
        
        return {
            'delta_current': current_delta_pct,
            'delta_cumulative_norm': cum_delta / abs(cum_delta).max() if cum_delta != 0 else 0,
            'delta_roc': delta_roc / df['volume'].mean() if df['volume'].mean() > 0 else 0,
            'delta_ma20': delta_ma,
            'delta_divergence': divergence
        }
```

### 4.4 Quando Usar

| Situação | Usar Delta? | Motivo |
|----------|-------------|--------|
| Entry Confirmation | ✅ SIM | Delta alinhado = entrada mais segura |
| Divergence Detection | ✅ SIM | Divergências podem indicar reversão |
| Breakout Validation | ✅ SIM | Breakout com delta = genuíno |
| Primary Signal | ❌ NÃO | Use SMC como primário |

---

## 5. 🔄 Imbalance Detection

### 5.1 O Que É

Imbalance = Desequilíbrio significativo entre compradores e vendedores.

Em footprint charts:
- **Stacked Imbalances**: Múltiplos níveis consecutivos com ratio > 3:1
- **Absorption**: Alto volume mas preço não move

### 5.2 Aproximação para Backtest

Podemos detectar padrões que INDICAM imbalance:

```python
def detect_imbalance_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta padrões que indicam imbalance institucional.
    """
    result = df.copy()
    
    # 1. Volume Spike (volume > 2x média)
    vol_ma = result['volume'].rolling(20).mean()
    result['volume_spike'] = result['volume'] > (vol_ma * 2)
    
    # 2. Rejection Candle (corpo pequeno, sombra grande, volume alto)
    body = abs(result['close'] - result['open'])
    range_size = result['high'] - result['low']
    body_ratio = body / range_size
    
    result['rejection_candle'] = (
        (body_ratio < 0.3) &  # Corpo < 30% do range
        (result['volume'] > vol_ma * 1.5)  # Volume acima da média
    )
    
    # 3. Absorption (preço parado mas volume alto)
    price_change = abs(result['close'].pct_change())
    result['absorption'] = (
        (price_change < 0.001) &  # Preço moveu < 0.1%
        (result['volume'] > vol_ma * 2)  # Volume 2x
    )
    
    # 4. Breakout with Volume
    high_20 = result['high'].rolling(20).max()
    low_20 = result['low'].rolling(20).min()
    
    result['breakout_up'] = (
        (result['close'] > high_20.shift(1)) &
        (result['volume'] > vol_ma * 1.5)
    )
    result['breakout_down'] = (
        (result['close'] < low_20.shift(1)) &
        (result['volume'] > vol_ma * 1.5)
    )
    
    return result
```

---

## 6. 📁 Estrutura de Arquivos Proposta

```
Python_Agent_Hub/
└── ml_pipeline/
    ├── indicators/                    # NOVO
    │   ├── __init__.py
    │   ├── volume_profile.py          # POC/VAH/VAL
    │   ├── volume_delta.py            # Delta from ticks/OHLCV
    │   └── imbalance_detector.py      # Imbalance patterns
    │
    ├── risk/                          # NOVO
    │   ├── __init__.py
    │   ├── r_multiple_tracker.py      # Van Tharp R-Multiple
    │   ├── risk_of_ruin.py            # Ralph Vince RoR
    │   └── position_sizing.py         # Kelly, Optimal f
    │
    ├── feature_engineering.py         # ATUALIZAR (adicionar novas features)
    ├── advanced_pipeline.py
    └── ... (existentes)
```

---

## 7. 📊 Novas Features para ML

### Features a Adicionar no `feature_engineering.py`:

| # | Feature | Fonte | Normalização |
|---|---------|-------|--------------|
| 16 | vp_poc_distance | Volume Profile | / ATR |
| 17 | vp_vah_distance | Volume Profile | / ATR |
| 18 | vp_val_distance | Volume Profile | / ATR |
| 19 | vp_in_value_area | Volume Profile | 0 ou 1 |
| 20 | delta_current | Volume Delta | já é -1 a 1 |
| 21 | delta_divergence | Volume Delta | -1, 0, 1 |
| 22 | imbalance_score | Imbalance | 0 a 1 |

---

## 8. 🎯 Checklist de Implementação

### Para Próximo Agente:

- [ ] Criar `Python_Agent_Hub/ml_pipeline/indicators/volume_profile.py`
- [ ] Criar `Python_Agent_Hub/ml_pipeline/indicators/volume_delta.py`
- [ ] Criar `Python_Agent_Hub/ml_pipeline/risk/r_multiple_tracker.py`
- [ ] Criar `Python_Agent_Hub/ml_pipeline/risk/risk_of_ruin.py`
- [ ] Atualizar `feature_engineering.py` com novas features
- [ ] Criar testes unitários para novos módulos
- [ ] Integrar com `ftmo_simulator.py`
- [ ] Documentar em `INDEX.md`

### Para Validação:

- [ ] Backtest com novas features
- [ ] Comparar WFE antes/depois
- [ ] Verificar se features são realmente preditivas
- [ ] Monte Carlo com Risk of Ruin < 5%

---

## 9. 📚 Referências

### Livros (no RAG):
- "Trade Your Way to Financial Freedom" - Van Tharp (R-Multiple, SQN)
- "The Mathematics of Money Management" - Ralph Vince (Risk of Ruin, Optimal f)
- "Market Profile" - Dalton (Volume Profile, POC, Value Area)
- "Markets in Profile" - Dalton (Auction Market Theory)
- "Order Flow Trading Setups" - Valtos (Footprint, Delta)

### Papers Acadêmicos:
- Lee & Ready (1991) - "Inferring Trade Direction from Intraday Data"
- Kyle (1985) - "Continuous Auctions and Insider Trading"

---

*Documento criado: 2024-11-29*
*Para uso por agentes e desenvolvedores do EA_SCALPER_XAUUSD*
