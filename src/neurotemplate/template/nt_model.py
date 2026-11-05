"""Modeling hooks. All optional; define one to change what it says. `nt docs <hook>` has each one's
signature and an example to paste. A trained run keeps a copy of this file."""

# def block_mapping(nm_mapping, cfg): ...       # split the model into parts: an RC zone, a thermostat, a meter
# def custom_models(): ...                      # bring your own model
# def output_heads(nm_mapping, norm_stats): ... # keep an output within limits
# def column_norm_config(nm_mapping): ...       # override a column's normalisation
# def synth_cfg(nm_mapping, norm_stats): ...    # the physics checks: what "at rest" means for your plant
# def sub_plot(df_true, column_name): ...       # how a channel is plotted
