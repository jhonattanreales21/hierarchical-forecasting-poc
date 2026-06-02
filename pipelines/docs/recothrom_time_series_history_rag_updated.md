# Recothrom Time Series History - RAG Knowledge Document

- **Purpose:** provide month-specific historical context so a Copilot Studio agent can explain demand drivers, anomalies, rebate effects, and uncertainty.

## Core business facts

- HCA is the only customer eligible for rebates in this context.
- Confirmed rebate-driven peaks occurred in October 2024 and July 2025.
- The July 2025 rebate was approximately 7%.
- Rebates are not invoice price discounts; they are post-purchase credit notes. Therefore, Average Sales Price may decline after the credit note effect even if invoice pricing did not change.
- A successful rebate is supported by higher volume and lower ASP. The attached Excel has demand and flags, but not ASP values; ASP confirmation comes from business context.
- Other rebate_target months may represent failed attempts by HCA to reach a target when ASP does not decline. These can still create artificial pull-forward demand.

## Data dictionary and semantic variable definitions

This section explains the meaning of each column in the forecasting dataset in natural business language. It is intended to help the Copilot Studio agent translate model variables into executive explanations. When answering user questions, the agent should use these definitions together with the month-specific timeline below.

### Date

- **Technical meaning:** Monthly period of the observation.
- **Business meaning:** Identifies the month being analyzed in the Recothrom demand history.
- **How to use in explanations:** Use this field to retrieve the specific month requested by the user and connect it with demand, calendar structure, rebate signals, and competitor supply signals.

### Demand

- **Technical meaning:** Monthly demand volume for Recothrom.
- **Business meaning:** Represents the monthly demand signal used for forecasting. Demand peaks may reflect true market consumption, customer buying behavior, pull-forward before rebate targets, competitor disruption, or other commercial events.
- **How to use in explanations:** Always interpret demand together with business drivers. A high month is not automatically true baseline growth; it may be artificial demand caused by rebate behavior or customer purchasing timing.

### pfizer_limited

- **Technical meaning:** Binary indicator where 1 means Pfizer was limited during the same month and 0 means no Pfizer limitation was identified.
- **Business meaning:** Pfizer is the main competitor. When Pfizer experiences supply issues, customers may shift demand toward Recothrom, creating incremental demand for our product.
- **Expected demand impact:** Usually positive. Same-month Pfizer supply disruption may increase Recothrom demand.
- **How to use in explanations:** If this variable is active, explain that competitor supply constraints may have supported demand uplift. Be clear that the mechanism is competitor availability, not a Recothrom price discount.

### pfizer_limited_lag1

- **Technical meaning:** One-month lag of the Pfizer limitation indicator.
- **Business meaning:** Captures the delayed demand response after Pfizer experiences supply disruption. Customers may not immediately switch demand in the exact same month; the impact can appear in the following month due to ordering cycles, contract behavior, inventory usage, or purchasing lead times.
- **Expected demand impact:** Usually positive with a one-month delay.
- **How to use in explanations:** If this variable is active, explain that demand may be responding to a prior-month Pfizer disruption rather than a current-month event.

### surgifoam_limited

- **Technical meaning:** Binary indicator where 1 means Surgifoam was limited and 0 means no limitation was identified.
- **Business meaning:** Surgifoam is another product/supply signal that may affect the market context for hemostatic products. A limitation can change customer behavior, product substitution dynamics, and purchasing patterns.
- **Expected demand impact:** Context-dependent. It may support or distort Recothrom demand depending on substitution behavior and product availability.
- **How to use in explanations:** If Surgifoam is limited (binary indicator equal to 1), then Baxter demand decreases. If Surgifoam is healthy (supply-wise), Baxter demand behaves under normal conditions.

### rebate_target

- **Technical meaning:** Binary indicator where 1 means a rebate target event or potential rebate-target behavior was identified.
- **Business meaning:** HCA is the only customer eligible for rebates in this context. When HCA is close to achieving a rebate threshold, it may increase purchases to reach the target. This can create artificial demand peaks or pull-forward demand.
- **Expected demand impact:** Usually positive in the event month because purchases may accelerate.
- **Important interpretation rule:** Confirmed successful rebate events are October 2024 and July 2025. Other months flagged as rebate_target may represent attempts by HCA to reach the rebate target but not necessarily successful rebates, especially if there is no evidence of ASP reduction.
- **How to use in explanations:** Distinguish between confirmed rebate events and possible failed attempts. Do not say every rebate_target month was a successful rebate.

### rebate_payback_lag1

- **Technical meaning:** One-month lag indicator after a rebate-target month.
- **Business meaning:** Captures the possible payback or normalization effect after a rebate-driven demand pull-forward. If customers accelerated purchases in the prior month, the following month may show lower demand because some future demand was already pulled forward.
- **Expected demand impact:** Usually negative or normalizing after a prior spike.
- **How to use in explanations:** If active, explain that demand may be lower because the previous month borrowed demand from the current month.

### bdays_m

- **Technical meaning:** Number of business days in the month.
- **Business meaning:** More business days generally create more opportunity for orders, shipments, and commercial activity. Fewer business days can constrain monthly volume even if underlying demand is stable.
- **Expected demand impact:** Usually positive when business days increase.
- **How to use in explanations:** Use this variable to contextualize whether a month had more or fewer selling/ordering days than usual.

### tue_total_m

- **Technical meaning:** Total number of Tuesdays in the month.
- **Business meaning:** Captures monthly calendar composition. Some demand patterns may be affected by weekday ordering or shipping routines.
- **Expected demand impact:** Context-dependent. A fifth Tuesday can increase ordering opportunities if Tuesday is an important operational day.
- **How to use in explanations:** Mention it when calendar structure may help explain changes in monthly demand.

### thu_total_m

- **Technical meaning:** Total number of Thursdays in the month.
- **Business meaning:** Captures monthly calendar composition related to Thursday ordering or shipping routines.
- **Expected demand impact:** Context-dependent. A fifth Thursday can increase ordering opportunities if Thursday is an important operational day.
- **How to use in explanations:** Use together with Thursday non-holiday and fifth-Thursday indicators to explain calendar effects.

### tue_nonhol_m

- **Technical meaning:** Number of Tuesdays in the month that are not holidays.
- **Business meaning:** Measures effective Tuesday ordering/shipping opportunities after removing holidays. This is more informative than total Tuesdays when holidays disrupt activity.
- **Expected demand impact:** Usually positive if Tuesdays are operationally important.
- **How to use in explanations:** If total Tuesdays are high but non-holiday Tuesdays are lower, explain that holidays may have reduced the effective calendar benefit.

### thu_nonhol_m

- **Technical meaning:** Number of Thursdays in the month that are not holidays.
- **Business meaning:** Measures effective Thursday ordering/shipping opportunities after removing holidays.
- **Expected demand impact:** Usually positive if Thursdays are operationally important.
- **How to use in explanations:** Use this variable to explain whether Thursday-related calendar opportunities were truly available or partially reduced by holidays.

### has_5th_tue_m

- **Technical meaning:** Binary indicator where 1 means the month has a fifth Tuesday.
- **Business meaning:** A fifth Tuesday can create an extra operational/order day compared with a typical month.
- **Expected demand impact:** Potentially positive if Tuesday is a meaningful ordering or shipment day.
- **How to use in explanations:** Explain it as an extra calendar opportunity, not as a business event.

### has_5th_thu_m

- **Technical meaning:** Binary indicator where 1 means the month has a fifth Thursday.
- **Business meaning:** A fifth Thursday can create an extra operational/order day compared with a typical month.
- **Expected demand impact:** Potentially positive if Thursday is a meaningful ordering or shipment day.
- **How to use in explanations:** Explain it as an extra calendar opportunity, not as a rebate or competitive event.

### tue_hol_m

- **Technical meaning:** Number of Tuesdays in the month that are holidays.
- **Business meaning:** Tuesday holidays can reduce effective selling, ordering, or shipping activity.
- **Expected demand impact:** Potentially negative if Tuesday is an important operational day.
- **How to use in explanations:** Use this to explain why total Tuesday count may overstate actual operational opportunity.

### thu_hol_m

- **Technical meaning:** Number of Thursdays in the month that are holidays.
- **Business meaning:** Thursday holidays can reduce effective selling, ordering, or shipping activity.
- **Expected demand impact:** Potentially negative if Thursday is an important operational day.
- **How to use in explanations:** Use this to explain why total Thursday count may overstate actual operational opportunity.

### ASP / Average Sales Price (business validation field)

- **Technical meaning:** Average sales price is not present as a column in the attached forecasting file, but it is a critical external validation signal.
- **Business meaning:** ASP helps confirm whether a demand peak was associated with a successful rebate. In this context, rebates are not invoice price discounts; they are post-purchase credit notes. However, a successful rebate may still appear as a lower ASP after the credit-note effect.
- **How to use in explanations:** For October 2024 and July 2025, the combination of volume increase and ASP decrease supports the conclusion that the peaks were true successful rebate events. For other rebate_target months, if ASP does not decline, describe them as possible failed attempts to reach a rebate target rather than confirmed rebates.

## Agent interpretation rules for variable-driven explanations

- Do not treat every demand peak as sustainable baseline growth. Some peaks may be artificial pull-forward demand.
- Confirmed successful rebate events: October 2024 and July 2025.
- The July 2025 rebate was approximately 7%, explaining the unusually high demand peak.
- Other rebate_target months should be described as possible HCA attempts to achieve a rebate target unless ASP evidence confirms a successful rebate.
- When rebate_payback_lag1 is active, explain possible post-rebate normalization or demand payback.
- When pfizer_limited or pfizer_limited_lag1 is active, explain the potential effect of Pfizer supply disruption on Recothrom demand.
- Calendar variables should be interpreted as opportunity/context variables, not as standalone causal events.
- Always separate confirmed business facts from model-informed hypotheses.
- If evidence is incomplete, state the uncertainty clearly.

## Month-by-month retrieval table


### 2023-01

- **Demand:** 588.14 scaled units
- **MoM:** N/A
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-02

- **Demand:** 824.09 scaled units
- **MoM:** +40.1%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 19 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-03

- **Demand:** 766.22 scaled units
- **MoM:** -7.0%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 23 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-04

- **Demand:** 633.39 scaled units
- **MoM:** -17.3%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-05

- **Demand:** 746.87 scaled units
- **MoM:** +17.9%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-06

- **Demand:** 1,065.61 scaled units
- **MoM:** +42.7%
- **YoY:** N/A
- **Signals:** rebate_target
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Rebate-target month in the model file. However, based on the business explanation, this may represent a possible HCA attempt to reach a rebate target rather than a confirmed successful rebate, unless ASP evidence confirms the credit effect. Treat as a potential artificial demand peak with medium/low confidence.
- **Confidence:** Medium

### 2023-07

- **Demand:** 663.63 scaled units
- **MoM:** -37.7%
- **YoY:** N/A
- **Signals:** rebate_payback_lag1
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 3/4.
- **Interpretation:** Post-rebate/payback lag month. Demand may be depressed or normalized after a prior pull-forward event. Interpret carefully because the previous month may have borrowed demand from this period.
- **Confidence:** Medium

### 2023-08

- **Demand:** 497.25 scaled units
- **MoM:** -25.1%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 23 business days; Tue/Thu non-holiday counts 5/5.
- **Interpretation:** Low-demand month versus the overall historical average. No confirmed rebate success is indicated. Potential explanations include post-event normalization, customer ordering timing, supply/customer behavior, or missing business context.
- **Confidence:** Low/Medium

### 2023-09

- **Demand:** 759.58 scaled units
- **MoM:** +52.8%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-10

- **Demand:** 721.75 scaled units
- **MoM:** -5.0%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2023-11

- **Demand:** 903.60 scaled units
- **MoM:** +25.2%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** High-demand month versus the overall historical average. No confirmed rebate flag is active, so this should be treated as elevated demand potentially driven by normal commercial activity, customer ordering behavior, calendar configuration, or unobserved business drivers.
- **Confidence:** Low/Medium

### 2023-12

- **Demand:** 766.44 scaled units
- **MoM:** -15.2%
- **YoY:** N/A
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-01

- **Demand:** 865.03 scaled units
- **MoM:** +12.9%
- **YoY:** +47.1%
- **Signals:** None
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-02

- **Demand:** 372.24 scaled units
- **MoM:** -57.0%
- **YoY:** -54.8%
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Low-demand month versus the overall historical average. No confirmed rebate success is indicated. Potential explanations include post-event normalization, customer ordering timing, supply/customer behavior, or missing business context.
- **Confidence:** Low/Medium

### 2024-03

- **Demand:** 574.61 scaled units
- **MoM:** +54.4%
- **YoY:** -25.0%
- **Signals:** None
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-04

- **Demand:** 677.04 scaled units
- **MoM:** +17.8%
- **YoY:** +6.9%
- **Signals:** None
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-05

- **Demand:** 843.21 scaled units
- **MoM:** +24.5%
- **YoY:** +12.9%
- **Signals:** rebate_target
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Rebate-target month in the model file. However, based on the business explanation, this may represent a possible HCA attempt to reach a rebate target rather than a confirmed successful rebate, unless ASP evidence confirms the credit effect. Treat as a potential artificial demand peak with medium/low confidence.
- **Confidence:** Medium

### 2024-06

- **Demand:** 741.90 scaled units
- **MoM:** -12.0%
- **YoY:** -30.4%
- **Signals:** rebate_payback_lag1
- **Calendar context:** 19 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Post-rebate/payback lag month. Demand may be depressed or normalized after a prior pull-forward event. Interpret carefully because the previous month may have borrowed demand from this period.
- **Confidence:** Medium

### 2024-07

- **Demand:** 517.39 scaled units
- **MoM:** -30.3%
- **YoY:** -22.0%
- **Signals:** None
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/3.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-08

- **Demand:** 747.23 scaled units
- **MoM:** +44.4%
- **YoY:** +50.3%
- **Signals:** None
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-09

- **Demand:** 809.18 scaled units
- **MoM:** +8.3%
- **YoY:** +6.5%
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2024-10

- **Demand:** 967.10 scaled units
- **MoM:** +19.5%
- **YoY:** +34.0%
- **Signals:** rebate_target
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/5.
- **Interpretation:** Confirmed rebate-driven demand peak. Business input indicates HCA obtained a rebate; this is consistent with a high volume month and a lower Average Sales Price caused by a post-purchase credit note. This should be treated as an artificial pull-forward / incentive-driven demand peak, not as a clean baseline demand signal.
- **Confidence:** High

### 2024-11

- **Demand:** 502.61 scaled units
- **MoM:** -48.0%
- **YoY:** -44.4%
- **Signals:** rebate_payback_lag1, surgifoam_limited
- **Calendar context:** 19 business days; Tue/Thu non-holiday counts 4/3.
- **Interpretation:** Post-rebate/payback lag month. Demand may be depressed or normalized after a prior pull-forward event. Interpret carefully because the previous month may have borrowed demand from this period.
- **Confidence:** Medium

### 2024-12

- **Demand:** 467.05 scaled units
- **MoM:** -7.1%
- **YoY:** -39.1%
- **Signals:** surgifoam_limited
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Surgifoam limited-supply flag is active. This may have altered customer purchasing behavior or mix, but the exact direction should be validated with commercial/supply notes. Avoid interpreting this month as purely organic demand.
- **Confidence:** Medium

### 2025-01

- **Demand:** 522.21 scaled units
- **MoM:** +11.8%
- **YoY:** -39.6%
- **Signals:** surgifoam_limited
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Surgifoam limited-supply flag is active. This may have altered customer purchasing behavior or mix, but the exact direction should be validated with commercial/supply notes. Avoid interpreting this month as purely organic demand.
- **Confidence:** Medium

### 2025-02

- **Demand:** 599.98 scaled units
- **MoM:** +14.9%
- **YoY:** +61.2%
- **Signals:** surgifoam_limited
- **Calendar context:** 19 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Surgifoam limited-supply flag is active. This may have altered customer purchasing behavior or mix, but the exact direction should be validated with commercial/supply notes. Avoid interpreting this month as purely organic demand.
- **Confidence:** Medium

### 2025-03

- **Demand:** 571.91 scaled units
- **MoM:** -4.7%
- **YoY:** -0.5%
- **Signals:** surgifoam_limited
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Surgifoam limited-supply flag is active. This may have altered customer purchasing behavior or mix, but the exact direction should be validated with commercial/supply notes. Avoid interpreting this month as purely organic demand.
- **Confidence:** Medium

### 2025-04

- **Demand:** 670.43 scaled units
- **MoM:** +17.2%
- **YoY:** -1.0%
- **Signals:** None
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2025-05

- **Demand:** 723.90 scaled units
- **MoM:** +8.0%
- **YoY:** -14.1%
- **Signals:** None
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2025-06

- **Demand:** 683.13 scaled units
- **MoM:** -5.6%
- **YoY:** -7.9%
- **Signals:** pfizer_limited
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/3.
- **Interpretation:** Pfizer limited-supply signal is active in the current month and/or lag. This may have supported incremental demand or substitution effects, depending on competitive availability. Interpret as exogenous market-condition impact rather than pure seasonality.
- **Confidence:** Medium

### 2025-07

- **Demand:** 1,084.63 scaled units
- **MoM:** +58.8%
- **YoY:** +109.6%
- **Signals:** rebate_target, pfizer_limited, pfizer_limited_lag1
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/5.
- **Interpretation:** Confirmed rebate-driven demand peak. HCA achieved an estimated 7% rebate in July 2025. The demand peak is consistent with incentive-driven purchasing and should be modeled with rebate_target=1. This is the strongest rebate event in the history provided.
- **Confidence:** High

### 2025-08

- **Demand:** 601.60 scaled units
- **MoM:** -44.5%
- **YoY:** -19.5%
- **Signals:** rebate_payback_lag1, pfizer_limited, pfizer_limited_lag1
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Post-rebate/payback lag month. Demand may be depressed or normalized after a prior pull-forward event. Interpret carefully because the previous month may have borrowed demand from this period.
- **Confidence:** Medium

### 2025-09

- **Demand:** 792.24 scaled units
- **MoM:** +31.7%
- **YoY:** -2.1%
- **Signals:** pfizer_limited, pfizer_limited_lag1
- **Calendar context:** 21 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Pfizer limited-supply signal is active in the current month and/or lag. This may have supported incremental demand or substitution effects, depending on competitive availability. Interpret as exogenous market-condition impact rather than pure seasonality.
- **Confidence:** Medium

### 2025-10

- **Demand:** 853.54 scaled units
- **MoM:** +7.7%
- **YoY:** -11.7%
- **Signals:** pfizer_limited, pfizer_limited_lag1
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 4/5.
- **Interpretation:** Pfizer limited-supply signal is active in the current month and/or lag. This may have supported incremental demand or substitution effects, depending on competitive availability. Interpret as exogenous market-condition impact rather than pure seasonality.
- **Confidence:** Medium

### 2025-11

- **Demand:** 827.74 scaled units
- **MoM:** -3.0%
- **YoY:** +64.7%
- **Signals:** pfizer_limited_lag1
- **Calendar context:** 18 business days; Tue/Thu non-holiday counts 3/3.
- **Interpretation:** Pfizer limited-supply signal is active in the current month and/or lag. This may have supported incremental demand or substitution effects, depending on competitive availability. Interpret as exogenous market-condition impact rather than pure seasonality.
- **Confidence:** Medium

### 2025-12

- **Demand:** 600.24 scaled units
- **MoM:** -27.5%
- **YoY:** +28.5%
- **Signals:** None
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/3.
- **Interpretation:** Relatively normal demand month compared with the available history. Use as part of baseline behavior unless additional business notes indicate a specific event.
- **Confidence:** Low/Medium

### 2026-01

- **Demand:** 497.79 scaled units
- **MoM:** -17.1%
- **YoY:** -4.7%
- **Signals:** None
- **Calendar context:** 20 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Low-demand month versus the overall historical average. No confirmed rebate success is indicated. Potential explanations include post-event normalization, customer ordering timing, supply/customer behavior, or missing business context.
- **Confidence:** Low/Medium

### 2026-02

- **Demand:** 644.84 scaled units
- **MoM:** +29.5%
- **YoY:** +7.5%
- **Signals:** pfizer_limited
- **Calendar context:** 19 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** Pfizer limited-supply signal is active in the current month and/or lag. This may have supported incremental demand or substitution effects, depending on competitive availability. Interpret as exogenous market-condition impact rather than pure seasonality.
- **Confidence:** Medium

## Compliance and Scaling Note

- Demand values in this Markdown version are expressed in transformed demand units using a confidential academic compliance factor.
- The original DOCX contains internal business quantities. This Markdown file should be preferred for repository, thesis, and RAG-development work unless internal values are explicitly required in a controlled environment.
- Percentage changes, flags, and interpretation rules are unchanged by the linear compliance transformation.

## Additional Scenario Variable Definitions

### pfizer_limited_lag2

- **Technical meaning:** Two-month lag of the Pfizer limitation indicator.
- **Business meaning:** Captures persistence of competitor supply disruption across the operational lead-time window.
- **Expected demand impact:** Usually positive when a Pfizer disruption continues to influence Recothrom demand beyond the first lag month.
- **How to use in explanations:** Use this variable to explain that the competitor supply effect may still be present even when the original Pfizer event started two months earlier.

### expected_market_share

- **Technical meaning:** Main planner-facing market-share assumption for the scenario.
- **Business meaning:** Encodes market intelligence about how much of the market Recothrom may serve when Pfizer is constrained.
- **Scenario values used:** 50% normal/default historical market share, 40% during Surgifoam-limited months, 65% during 2025 Pfizer limited-supply impact months, 70% in March 2026, 75% in April 2026, 80% in May 2026, 100% from June to December 2026 under the severe Pfizer stockout scenario, and 65% in 2027 as a retained-customer baseline after the disruption.
- **How to use in explanations:** Treat this as the primary scenario-intensity input. If a user asks why the forecast increases, explain the expected market-share assumption first.

### market_share_uplift_for_model

- **Technical meaning:** Backend feature calculated as `max(expected_market_share - 0.50, 0)`.
- **Business meaning:** Measures the simple incremental share above the normal 50% baseline.
- **Examples:** 65% share becomes 0.15, 75% becomes 0.25, 80% becomes 0.30, and 100% becomes 0.50.
- **How to use in explanations:** Mention only when explaining model mechanics. It is not entered manually by planners.

### market_share_stress_for_model

- **Technical meaning:** Backend nonlinear transformation of `expected_market_share`.
- **Values used:** 50% maps to 0.00, 65% maps to 0.14, 75% maps to 1.00, 80% maps to 1.55, and 100% maps to 3.00. Intermediate values are interpolated.
- **Business meaning:** Represents the idea that moving from 75% to 100% expected share is a major competitor-stockout regime change, not a small incremental increase.
- **How to use in explanations:** Mention only when explaining why Prophet reacts strongly to May 2026 onward. For business users, translate it as: higher expected market share creates stronger nonlinear forecast pressure.

## Additional Agent Interpretation Rules for Market Share and Pfizer Context

- Expected market share is the primary scenario variable for May 2026 onward.
- Pfizer limited supply has a delayed impact on Recothrom demand. The one-month lag is the most important timing signal, while the two-month lag captures persistence.
- Do not say the model learned the 2026 severe-stockout regime fully from history. The future regime is represented through planner-supplied expected market share and backend transformations.
- For March and April 2026, explain the demand increase as a combination of active Pfizer limitation, lagged Pfizer effect, and expected market-share uplift.
- For May 2026, explain values as a transition scenario with 80% expected market share. For June to December 2026, explain values as forecasts under a 100% expected-share severe competitor-stockout scenario, not as actual demand.
- When answering executives, state whether a statement is a confirmed historical fact, a model feature, or a forward-looking scenario assumption.

## Added Historical Retrieval Entries

### 2026-03

- **Demand:** 908.02 scaled units
- **MoM:** +40.8%
- **YoY:** +58.8%
- **Signals:** pfizer_limited, pfizer_limited_lag1, expected_market_share=0.70
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 5/4.
- **Interpretation:** Confirmed 2026 Pfizer limited-supply regime. Demand increased materially versus February 2026 and versus March 2025. The agent should explain this as a business-disruption month where Pfizer availability problems and lagged competitor effects likely supported Recothrom demand uplift. Expected market share is a scenario assumption used to represent approximately 70% share capture.
- **Confidence:** High for the observed demand and active Pfizer signal; Medium for the exact market-share estimate.

### 2026-04

- **Demand:** 962.72 scaled units
- **MoM:** +6.0%
- **YoY:** +43.6%
- **Signals:** pfizer_limited, pfizer_limited_lag1, pfizer_limited_lag2, expected_market_share=0.75
- **Calendar context:** 22 business days; Tue/Thu non-holiday counts 4/4.
- **Interpretation:** April 2026 is the operational backtesting target used in the thesis because May 2026 actual demand was not yet available. Demand remained elevated and is consistent with a continued Pfizer limited-supply environment. The expected market-share assumption is 75%, representing stronger share capture than March but not yet the severe-stockout scenario used from May onward.
- **Confidence:** High for observed demand and Pfizer limitation; Medium for the exact market-share estimate.

## Forward-Looking Scenario Retrieval Table

These rows are forecast scenario records, not actual demand. They are intended for the conversational layer to explain May 2026 onward planning assumptions.

| Month | Expected Market Share | Pfizer Context Label | Prophet Scenario Forecast | CatBoost Scenario Forecast | Interpretation |
|---|---:|---|---:|---:|---|
| 2026-05 | 80% | Medium disruption | 1,097.19 | 905.09 | Transition scenario; Prophet reflects the nonlinear market-share stress curve and lands near the expected May planning level. |
| 2026-06 | 100% | Severe stockout | 1,451.74 | 877.24 | Severe Pfizer stockout scenario; Prophet reacts to the nonlinear 100% share stress. |
| 2026-07 | 100% | Severe stockout | 1,449.26 | 843.76 | Severe Pfizer stockout scenario; Prophet remains in the 1,400-1,500 planning range. |
| 2026-08 | 100% | Severe stockout | 1,446.70 | 794.90 | Severe Pfizer stockout scenario; CatBoost remains conservative relative to Prophet. |
| 2026-09 | 100% | Severe stockout | 1,444.14 | 860.65 | Severe Pfizer stockout scenario; Prophet remains aligned with the expected high-share planning regime. |
| 2026-10 | 100% | Severe stockout | 1,441.67 | 870.94 | Severe Pfizer stockout scenario; Prophet remains the primary scenario engine. |
| 2026-11 | 100% | Severe stockout | 1,439.11 | 862.94 | Severe Pfizer stockout scenario; Prophet remains the primary scenario engine. |
| 2026-12 | 100% | Severe stockout | 1,436.63 | 840.81 | Severe Pfizer stockout scenario; Prophet remains in the 1,400-1,500 planning range. |
| 2027-01 | 65% | Retained baseline | 727.66 | 773.03 | Retained-customer baseline after the severe stockout planning window. |
| 2027-02 | 65% | Retained baseline | 574.60 | 674.43 | Retained-customer baseline after the severe stockout planning window. |
| 2027-03 | 65% | Retained baseline | 621.21 | 678.46 | Retained-customer baseline after the severe stockout planning window. |
| 2027-04 | 65% | Retained baseline | 618.65 | 678.48 | Retained-customer baseline after the severe stockout planning window. |

## RAG Readiness Improvement Recommendations

The document is suitable as a first knowledge base for the conversational layer, but the following practices should be applied when loading it into a RAG system:

- **Chunk by month and variable definition:** Keep each monthly record as an independent retrievable chunk. Keep variable definitions as separate chunks so the agent can retrieve both the month and the meaning of the active signals.
- **Preserve confidence labels:** The agent should surface confidence when explaining a month. High-confidence statements can be presented directly; Medium or Low/Medium statements should be framed as plausible interpretations.
- **Separate fact, feature, and scenario:** Historical demand and confirmed rebate events are facts. Model flags are engineered features. May 2026 onward values are forecast scenarios, not actual demand.
- **Use transformed demand units in external-facing responses:** Unless the user is explicitly operating in a controlled internal environment, the agent should answer using transformed units from this Markdown file.
- **Avoid unsupported causal certainty:** The agent should not say Pfizer disruption, rebates, or calendar effects definitively caused a demand movement unless the month has high-confidence supporting business evidence.
- **Prioritize expected market share for 2026 explanations:** For March 2026 onward, the agent should explain expected-market-share assumptions first, then use Pfizer limited-supply context as supporting business narrative.
- **Connect explanations to planning decisions:** When asked about May 2026 onward, the agent should describe the business scenario, the Prophet scenario forecast, and why alternative models remain flatter.
