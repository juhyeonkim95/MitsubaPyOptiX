def edit_compile_config(use_nee, sampling_strategy):
	with open("app_config.h", 'w') as f:
		lines = [
			"#ifndef APP_CONFIG_H",
			"#define APP_CONFIG_H",
			"#define SAMPLING_STRATEGY_BSDF 0",
			"#define SAMPLING_STRATEGY_SD_TREE 1",
			"#define USE_NEXT_EVENT_ESTIMATION %d\n" % (1 if use_nee else 0),
			"#define SAMPLING_STRATEGY %s\n" % sampling_strategy,
			"#endif"
		]
		f.write('\n'.join(lines))
