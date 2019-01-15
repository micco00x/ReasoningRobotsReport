breakoutfull_name2robot_feature_ext = {
    "N": BreakoutNRobotFeatureExtractor,
    "S": BreakoutSRobotFeatureExtractor
}

breakoutfull_name2temp_goals = {
    "cols": [BreakoutCompleteColumnsTemporalEvaluator],
    "rows": [BreakoutCompleteRowsTemporalEvaluator],
    "both": [BreakoutCompleteColumnsTemporalEvaluator, BreakoutCompleteRowsTemporalEvaluator]
}


def _set_up_temporal_breakout(config, args, env, robot_feature_extractor, brain):
    temporal_goals = []
    if args.temp_goal == "cols" or args.temp_goal == "both":
        by_cols = BreakoutCompleteColumnsTemporalEvaluator(env.observation_space, bricks_rows=args.brick_rows,
                                                           bricks_cols=args.brick_cols, left_right=args.left_right,
                                                           gamma=config.gamma, on_the_fly=config.on_the_fly)
        temporal_goals.append(by_cols)

    if args.temp_goal == "rows" or args.temp_goal == "both":
        by_rows = BreakoutCompleteRowsTemporalEvaluator(env.observation_space, bricks_rows=args.brick_rows,
                                                        bricks_cols=args.brick_cols, bottom_up=args.bottom_up,
                                                        gamma=config.gamma, on_the_fly=config.on_the_fly)
        temporal_goals.append(by_rows)

    agent = TGAgent(robot_feature_extractor,
                    brain,
                    temporal_goals,
                    reward_shaping=config.reward_shaping)

    tr = TGTrainerExt(env, agent, n_episodes=config.episodes,
                   stop_conditions=(GoalPercentage(100, 1.0),),
                   data_dir=config.datadir
                   )
    return agent, tr


def _set_up_simple_breakout(config, args, env, robot_feature_extractor, brain):
    agent = RLAgent(robot_feature_extractor, brain)
    tr = GenericTrainer(env, agent, n_episodes=config.episodes,
                        stop_conditions=(GoalPercentage(100, 1.0),),
                        data_dir=config.datadir)
    return agent, tr


def run_experiment(config:Config, args):
    env = gym.make("BreakoutNoFrameskip-v4")
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    render = config.render

    if config.resume:
        trainer = GenericTrainer if args.temp_goal is None else TGTrainerExt
        stats, optimal_stats = trainer.resume(render=render, verbosity=args.verbosity)
    elif config.eval:
        trainer = GenericTrainer if args.temp_goal is None else TGTrainerExt
        stats, optimal_stats = trainer.eval(render=render, verbosity=args.verbosity)
    else:
        robot_feature_extractor = breakoutfull_name2robot_feature_ext[args.robot_feature_space](env.observation_space)
        brain = name2algorithm[config.algorithm](None, env.action_space, policy=EGreedy(epsilon_start=1.0, decay_steps=10000000),
                                         alpha=config.alpha, gamma=config.gamma, lambda_=config.lambda_)
        if args.temp_goal is None:
            print("No temporal goal - simple Breakout")
            agent, trainer = _set_up_simple_breakout(config, args, env, robot_feature_extractor, brain)
        else:
            agent, trainer = _set_up_temporal_breakout(config, args, env, robot_feature_extractor, brain)

        stats, optimal_stats = trainer.main(render=render, verbosity=args.verbosity)

    return stats, optimal_stats
