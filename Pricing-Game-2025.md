# Pricing Game 2025

## Context

In this edition of the Pricing Game, you are the actuary of an insurance company competing in a motor Third-Party Liability insurance market.

- All participating teams make up 100% of the market share, and each is randomly assigned a portfolio of customers.
- The market overall is unprofitable, with a loss ratio above 100%.
- Existing policyholders first receive a renewal premium offer from their current insurer. If they decline, they get quotes from every company.
- There are minimum regulatory prices in place to avoid insolvencies  

Game structure:

1. Technical Pricing - Derive a technical rating structure based on GLMs to improve profitability.
2. Renewal Pricing – Refine the technical rating structure based on renewal premium increases to retain insureds.
3. New Business – Quote insureds that leave other companies to gain market share while mantaining profitable business.

## Market Report

This table shows the baseline position of each player entering the competition. Market share reflects **in-force policies** (current portfolio), while loss ratio shows **previous year performance** (full portfolio including cancelled policies).

| Player | Policies In-Force | Avg Earned | Pure Premium | Market Share | Loss Ratio | LR vs Market |
|-------:|------------------:|-----------:|-------------:|-------------:|-----------:|-------------:|
| 1 | 31,014 | $527.52 | $581.58 | 11.5% | 104.9% | -0.3pp |
| 2 | 27,225 | $505.36 | $579.48 | 10.1% | 108.8% | +3.6pp |
| 3 | 27,862 | $527.80 | $595.44 | 10.3% | 107.2% | +2.0pp |
| 4 | 30,103 | $525.23 | $578.60 | 11.1% | 104.7% | -0.5pp |
| 5 | 29,248 | $545.23 | $585.83 | 10.8% | 102.2% | -3.0pp |
| 6 | 28,567 | $531.23 | $590.19 | 10.6% | 105.5% | +0.3pp |
| 7 | 25,959 | $502.52 | $550.84 | 9.6% | 104.1% | -1.1pp |
| 8 | 24,309 | $568.29 | $615.74 | 9.0% | 102.9% | -2.3pp |
| 9 | 28,039 | $493.62 | $547.71 | 10.4% | 105.5% | +0.3pp |
| 10 | 17,867 | $516.29 | $583.53 | 6.6% | 107.4% | +2.2pp |

**Market Summary:**
- Total In-Force Policies: 270,193
- Average Market Share: 10.0% (10 players)
- Market Share Range: 6.6% - 11.5%
- Market LR: 105.2%
- LR Range: 102.2% - 108.8%


## Competition Scoring

Final Performance:
- Profitability Ranking: 3/2/1/0 points (1st/2nd/3rd/4th+ place by lowest loss ratio)

Improvement Achievement Score:
- Relative LR Improvement: +1/-1 points per 1pp improvement/deterioration in position vs market average
  - Measures change in: (Player LR - Market LR)
  - Positive score = improved position relative to market
- Market Share Improvement: +1/-1 point per 1% market share growth/shrink, with a low cap of -5 points 

## Goals

- Focus on making your own portfolio profitable
- Do not lose significant amount of business, i.e., avoid retention levels below 60%
- Consider your initial position in the market before launching your strategy

## Data Description

Each team receives a `loss_data.csv` file containing their assigned portfolio of motor TPL policies, a `rating_structure.json` that contains its current rating structure applied for existing customers, and a `retention.json` that contains the parameters of a logistic regression model to calculate renewal probabilities.

### Data (`loss_data.csv`)

#### Policy Information
- **policy_id**: Unique identifier for each policy
- **effective_date** / **expiration_date**: Policy period dates
- **cancellation_date**: Date of early cancellation (if applicable)
- **in_force**: Boolean flag indicating active policies (excludes historical cancellations)
- **exposure**: Policy exposure (typically fractional based on policy duration)
- **player_id**: Team assignment identifier
- **commercial_premium**: Current premium charged to the customer

#### Loss Information
- **claim_counts**: Number of claims per policy
- **loss_amt**: Total loss amount per policy

#### Risk Variables

**Ordinal Variables** (continuous ranges):
- **driver_age**: Driver age (18-100 years)
- **vehicle_age**: Vehicle age (0-30 years) 
- **horsepower**: Engine horsepower (40-300 HP)

**Nominal Variables** (categorical levels):
- **province**: Geographic region (13 provinces: RI, NJ, SH, MK, AS, JZ, BA, MD, QS, HA, HS, JF, TB)
- **vehicle_make**: Vehicle manufacturer (Toyota, Chevrolet, Ford, Hyundai, Isuzu, KIA, Mercedes, other)
- **body_type**: Vehicle type (sedan, heavycomm, lightcomm, pickup, 4x4)
- **vehicle_color**: Vehicle color (white, black, blue, silver, grey, red, other)
- **vehicle_transmission**: Transmission type (manual, automatic)

### Rating Structure (`rating_structure.json`)

#### Structure Format
```json
{
  "base_values": {
    "value": 525.03
  },
  "relativities": {
    "driver_age": {
      "18-25": 1.2500,
      "26-35": 1.0000,
      "36-50": 0.9500,
      "51-inf": 0.8500
    },
    "province": {
      "RI": 1.0000,
      "NJ": 1.1200,
      "SH": 0.9800
    }
  },
  "base_group": "driver_age: 26-35, province: RI, ..."
}
```

#### Premium Calculation Examples

**Example 1: Base Group**
- Driver age: 28 years (26-35 group) → Relativity: 1.0000
- Province: RI → Relativity: 1.0000
- **Premium Calculation**: $525.03 × 1.0000 × 1.0000 = **$525.03**

**Example 2: High-Risk Profile**
- Driver age: 22 years (18-25 group) → Relativity: 1.2500  
- Province: NJ → Relativity: 1.1200
- **Premium Calculation**: $525.03 × 1.2500 × 1.1200 = **$735.04**

## Submission Guidelines

- Team 'x' is required to submit a single rating structure file (`player_x.json`), which is generated by the `workflow.qmd`. This file will be used to price both renewal and new policies.

---
