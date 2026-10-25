# How this project works

Colour key: **blue** = external data or market event · **lavender** = code step · **red** = check, gate or decision · **green** = good outcome · **yellow** = caution · **grey** = output

## 1. Pipeline (what `make all` runs)

```mermaid
flowchart TD
    A["SOURCE: earnings calendar (announcement dates, 2023–2026)<br/>mirror: huggingface.co/datasets/defeatbeta/yahoo-finance-data"]
    B["SOURCE: reported quarterly EPS + shares outstanding"]
    C["SOURCE: daily prices and volume (stocks + SPY)"]
    I{"ingest.py<br/>sha256 matches config?"}
    A --> I
    B --> I
    C --> I
    I -- "no" --> X["Stop: source changed"]
    I -- "yes" --> L

    subgraph L["Build events · pipeline.py"]
      L1["Day 0 = first trading day on/after the announcement"]
      L2["SUE = (EPS − EPS a year ago) ÷ std of past 4–8 changes"]
      L3["EAR = return days −1..+1 minus SPY"]
      L4["Outcome: return days +2..+61 minus SPY"]
    end

    subgraph G["Gates"]
      G1{"Price ≥ $5 and $1M daily volume at day −2"}
      G2{"Quintile breakpoints from the prior 91 days only"}
      G3{"Unit tests: later EPS or later announcements<br/>cannot change an earlier signal"}
    end

    subgraph T["Tests"]
      T1["Top minus bottom quintile, by size,<br/>t-stat clustered by quarter"]
      T2["2023–24 vs 2025–26"]
      T3["Calendar-time long-short, costs 10/25/50 bp"]
    end

    O["reports/metrics.json · events.csv · figures/*.png"]
    L --> G --> T --> O
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef step fill:#ece8ff,stroke:#6b5fbf,color:#1d1640
    classDef gate fill:#ffe0e0,stroke:#c03030,color:#400000
    classDef out fill:#eeeeee,stroke:#777,color:#222
    class A,B,C src
    class L1,L2,L3,L4,T1,T2,T3 step
    class I,X,G1,G2,G3 gate
    class O out
```

![Pipeline](../reports/figures/flow-1.png)

## 2. The signal and the book decision

```mermaid
flowchart LR
    E["Earnings released"] --> R["Days −1..+1:<br/>most of the price move"]
    R --> F["Close of day +1:<br/>signal known, position formed"]
    F --> S{"Earnings surprise quintile"}
    S -- "best" --> L["Long: drifted 1.7 pp<br/>ahead of the worst over 60 days"]
    S -- "worst" --> SH["Short"]
    L --> B{"Book decision"}
    SH --> B
    B --> B1["Mid + large caps: 6.0% a year<br/>after 10 bp, Sharpe 0.93"]
    B --> B2["Costs above ~30 bp<br/>erase most of it"]
    B --> B3["Fading: 2.6 pp in 2023–24,<br/>0.5 pp in 2025–26"]
    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef gate fill:#ffe0e0,stroke:#c03030,color:#400000
    classDef ok fill:#e3f3e3,stroke:#3a8a3a,color:#103010
    classDef rule fill:#fff4cc,stroke:#b38f00,color:#3d3000
    class E,R,F src
    class S,B gate
    class L,B1 ok
    class SH,B2,B3 rule
```

![Decision](../reports/figures/flow-2.png)
