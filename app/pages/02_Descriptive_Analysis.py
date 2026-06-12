from ui.page_blocks.descriptive_blocks import (
    render_descriptive_analysis,
    render_descriptive_page_header,
)
from ui.styles import apply_global_styles
from utils.data_loaders import (
    load_demand_daily,
    load_demand_monthly_primary,
    load_demand_weekly,
    load_exogenous_monthly_primary,
)

apply_global_styles()
render_descriptive_page_header()

demand_by_granularity = {
    "daily": load_demand_daily(),
    "weekly": load_demand_weekly(),
    "monthly": load_demand_monthly_primary(),
}
exogenous = load_exogenous_monthly_primary()

render_descriptive_analysis(demand_by_granularity, exogenous)
