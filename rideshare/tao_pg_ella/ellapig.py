#----------------------------------------------------------------------------------------------------------------------#
# Title: Ellapig Command Line Interface
# Description: This file contains the command line interface functions for operating training and testing the Ellapig network(s).
# Author: Matthew Sentell
# Version: 0.02.21
# Last updated on: 2024.03.19
#----------------------------------------------------------------------------------------------------------------------#

from rideshare.tao_pg_ella.ellapig_training import *
import argparse

parser = argparse.ArgumentParser(description='Ellapig Trainer: performs one of the available commands, then stores and graphs the loss data')
parser_commands = parser.add_subparsers(help='commands', dest='cmd')
parser_training = parser_commands.add_parser('train', help='training commands').add_subparsers(help='trainers', dest='trainer')

parser_training_base = parser_training.add_parser('base', help='base learner training')
parser_training_base_exp = parser_training_base.add_argument_group('experiment')
parser_training_base_exp.add_argument('-b', '--batch', type=int, metavar='batches', help='Sets the number of batches over which results will be averaged.')
parser_training_base_exp.add_argument('-i', '--iter', type=int, default=500, metavar='iterations', dest='iterations', help='Sets the number of iterations the algorithm will process.')
parser_training_base_exp.add_argument('-e', '--epi', type=int, default=10, metavar='episodes', dest='episodes', help='Sets the number of episodes in each learning iteration.')
parser_training_base_exp.add_argument('-s', '--step', type=int, default=500, metavar='steps', help='Sets the maximum number of steps in each episode.')
parser_training_base_exp.add_argument('-ld', '--lat-dim', type=int, default=32, dest='dim', metavar='dimension', help='Sets the latent dimension of the networks.')
parser_training_base_exp.add_argument('-ly', '--layers', type=int, default=0, metavar='layers', help='Sets the number of hidden layers in the policy.')
parser_training_base_exp.add_argument('-lr', '--learn-rate', type=float, default=1e-2, metavar='rate', help='Sets the learning rate of the optimizer.')

parser_training_base_ext = parser_training_base.add_argument_group('extra')
parser_training_base_ext.add_argument('-n', '--notify', type=int, default=10, metavar='notifications', help='Sets the minimum number of partial-progress notifications.')
parser_training_base_ext.add_argument('-d', '--res-dir', type=str, metavar='directory', help='Sets the directory in which results will be stored.')
parser_training_base_ext.add_argument('-r', '--res-id', type=str, metavar='identifier', help='Sets the identifier for the results file(s).')

parser_training_cart = parser_training.add_parser('cart', help='CartPole training')
parser_training_cart_exp = parser_training_cart.add_argument_group('experiment')
parser_training_cart_exp.add_argument('-i', '--iter', type=int, default=[100], nargs='*', metavar='iterations', dest='iterations', help='Sets the number of iterations the algorithm will process. Multiple arguments will result in multiple runs for each provided number of iterations.')
parser_training_cart_exp.add_argument('-e', '--epi', type=int, default=10, metavar='episodes', dest='episodes', help='Sets the number of episodes in each learning iteration.')
parser_training_cart_exp.add_argument('-s', '--step', type=int, default=500, metavar='steps', help='Sets the maximum number of steps in each episode.')
parser_training_cart_exp.add_argument('-ld', '--lat-dim', type=int, default=32, metavar='size', help='Sets the size of the shared latent space dimension.')
parser_training_cart_exp.add_argument('-lt', '--tsk-dim', type=int, default=8, metavar='size', help='Sets the size of task-specific latent spaces (and that of the corresponding dimension of the shared latent space).')
parser_training_cart_exp.add_argument('-ly', '--layers', type=int, default=0, metavar='layers', help='Sets the number of hidden layers in the policy network.')
parser_training_cart_exp.add_argument('-lr', '--learn-rate', type=float, default=1e-3, metavar='rate', help='Sets the learning rate of the internal optimizers.')
parser_training_cart_exp.add_argument('-ls', '--learn-step', type=int, default=100, metavar='steps', help='Sets the number of steps to run interal optimizers for each task.')
parser_training_cart_exp.add_argument('-d', '--discount', type=float, default=9e-1, metavar='rate', help='Sets the discount rate of later steps in value calculations.')
parser_training_cart_exp.add_argument('-t', '--task', type=int, default=1, metavar='tasks', dest='tasks', help='Sets the simulated number of tasks to learn.')
parser_training_cart_exp.add_argument('--double', action='store_true', help='Enables the Marmotellpig implementation of the critic.')

parser_training_cart_ext = parser_training_cart.add_argument_group('extra')
parser_training_cart_ext.add_argument('-n', '--notify', action='store_false', help='Disables task-specific partial-progress notifications.')
parser_training_cart_ext.add_argument('-f', '--fixed', action='store_true', help='Fixes the trajectories over which learning is performed.')
parser_training_cart_ext.add_argument('-p', '--plot-partial', type=str, nargs='?', const='', metavar='identifier', help='Plots the last task-specific internal learning into a temporary location, overwriting previous versions.')

parser_training_ride = parser_training.add_parser('ride', help='Rideshare training')
parser_training_ride_exp = parser_training_ride.add_argument_group('experiment')
parser_training_ride_exp.add_argument('-i', '--iter', type=int, default=[10000], nargs='*', metavar='iterations', dest='iterations', help='Sets the number of iterations the algorithm will process. Multiple arguments will result in multiple runs for each provided number of iterations.')
parser_training_ride_exp.add_argument('-e', '--epi', type=int, default=1, metavar='episodes', dest='episodes', help='Sets the number of episodes in each learning iteration.')
parser_training_ride_exp.add_argument('-s', '--step', type=int, default=100, metavar='steps', help='Sets the maximum number of steps in each episode.')
parser_training_ride_exp.add_argument('-ld', '--lat-dim', '--sls', type=int, default=32, metavar='size', help='Sets the size of the shared latent space dimension.')
parser_training_ride_exp.add_argument('-lt', '--tsk-dim', '--tls', type=int, default=8, metavar='size', help='Sets the size of task-specific latent spaces (and that of the corresponding dimension of the shared latent space).')
parser_training_ride_exp.add_argument('-ly', '--layers', type=int, default=0, metavar='layers', help='Sets the number of hidden layers in the policy network.')
parser_training_ride_exp.add_argument('-lr', '--learn-rate', type=float, default=1e-2, metavar='rate', help='Sets the learning rate of the internal optimizers.')
parser_training_ride_exp.add_argument('-ls', '--learn-step', type=int, default=100, metavar='steps', help='Sets the number of steps to run interal optimizers for each task.')
parser_training_ride_exp.add_argument('-d', '--discount', type=float, default=9e-1, metavar='rate', help='Sets the discount rate of later steps in value calculations.')
parser_training_ride_exp.add_argument('-r', '--raw', action='store_true', help='Tells the trainer to learn from raw action trajectories instead of actual action trajectories.')
parser_training_ride_exp.add_argument('--double', action='store_true', help='Enables the Marmotellpig implementation of the critic.')
parser_training_ride_exp.add_argument('--decomp', '--decompose', action='store_true', dest='decomposed', help='Enables the Marmotellpig implementation of the critic.')
parser_training_ride_exp.add_argument('--clean', action='store_true', help='Enables expert-trajcetory training.')

parser_training_ride_env = parser_training_ride.add_argument_group('environment')
parser_training_ride_env.add_argument('-a', '--agents', type=int, default=3, metavar='agents', help='Sets the number of agents in the environment.')
parser_training_ride_env.add_argument('-g', '--grid', type=int, nargs=2, default=[5, 5], metavar=('length', 'width'), help='Sets the grid size for the environment.')
parser_training_ride_env.add_argument('-c', '--costs', type=float, nargs=6, default=[0, -0.1, -1.2, -2, None, 0], metavar=('accept', 'pickup', 'move', 'noop', 'dropoff', 'passenger-free'), help='Sets the various costs for the environment.')
parser_training_ride_env.add_argument('-t', '--task', type=int, default=2, metavar='tasks', help='Sets number of initial tasks in the environment.')
parser_training_ride_env.add_argument('-o', '--open', action='store_true', help='Sets environment to be task-open.')

parser_training_ride_ext = parser_training_ride.add_argument_group('extra')
parser_training_ride_ext.add_argument('-n', '--notify', action='store_false', help='Disables task-specific partial-progress notifications.')
parser_training_ride_ext.add_argument('-f', '--fixed', action='store_true', help='Fixes the trajectories over which learning is performed.')
parser_training_ride_ext.add_argument('-pt', '--plot-tasks', type=str, nargs='?', const='', metavar='identifier', help='Plots the last task-specific internal learning into a temporary location, overwriting previous versions.')
parser_training_ride_ext.add_argument('-pp', '--plot-partial', type=float, nargs='?', const=0.1, metavar='frequency', help='Plots the training run\'s progress according to the provided frequency.')

parser_training_resume = parser_training.add_parser('resume', help='resume a previous training session that was cut short')
# parser_training_resume.add_argument('func', type=str, nargs='?', default='ride', metavar='trainer', help='Sets the source folder of the configuration file and any policy or loss files tat may exist.')
parser_training_resume.add_argument('source', type=str, help='Sets the source folder of the configuration file and any policy or loss files tat may exist.')
parser_training_resume.add_argument('-e', '--extend', type=int, nargs='*', metavar='iterations', dest='extension', help='Sets a new iteration count for the resumed training. (May change save location!)')
parser_training_resume.add_argument('-p', '-pp', '--plot-partial', type=float, metavar='frequency', help='Updates the frequency at which training run\'s progress is plotted.')

parser_run = parser_commands.add_parser('run', help='Run a number of episodes without learning, using provided policies.')
parser_run.add_argument('sources', type=str, nargs='*', metavar='src', help='Sets the policy directory.')
parser_run.add_argument('-a', '--agents', type=int, nargs='*', default=[], help='Sets the number of random-policy agents.')
parser_run.add_argument('-e', '--epi', type=int, default=1000, metavar='episodes', dest='episodes', help='Sets the number of episodes in each learning iteration.')
parser_run.add_argument('-s', '--step', type=int, default=100, metavar='steps', dest='steps', help='Sets the maximum number of steps in each episode.')
parser_run.add_argument('-g', '--grid', type=int, nargs=2, default=[5, 5], metavar=('length', 'width'), help='Sets the grid size for the environment.')
parser_run.add_argument('-c', '--costs', type=float, nargs=6, default=[0, -0.1, -1.2, -2, None, 0], metavar=('accept', 'pickup', 'move', 'noop', 'dropoff', 'passenger-free'), help='Sets the various costs for the environment.')
parser_run.add_argument('-t', '--task', type=int, default=2, metavar='tasks', help='Sets number of initial tasks in the environment.')
parser_run.add_argument('-o', '--open', action='store_true', help='Sets environment to be task-open.')
parser_run.add_argument('--decomp', action='store_true', help='Enables decompoed policy for random or expert agents.')
parser_run.add_argument('--clean', action='store_true', help='Enables expert-trajectory initialization.')
parser_run.add_argument('--expert', action='store_true', help='Enables expert-trajectory initialization and policy for otherwise random agents.')
parser_run.add_argument('--store', action='store_true', help='Enables the storing of trajectory data on completion. This may consume significant storage space, especially on longer runs.')
parser_run.add_argument('-f', '--fixed', action='store_true', help='Fixes the starting point for each run.')
parser_run.add_argument('-n', '--notify', type=float, nargs='?', const=None, default=1, metavar='frequency', help='Sets the frequency of partial-progress notifications.')

parser_plot = parser_commands.add_parser('plot', help='Replots data from provided .pkl file(s) of the same type.')
parser_plot.add_argument('plotter', type=str, choices=['base', 'task', 'full', 'runs'], help='Sets the type of plotter function to use.')
parser_plot.add_argument('source', type=str, metavar='src', help='Sets the datafile to plot.')
parser_plot.add_argument('dest', type=str, nargs='?', help='Sets the target location of the plotted file.')
parser_plot.add_argument('-e', '--episodic', type=int, nargs='?', const=60, help='Tells the plotter to convert episodic rewards to immediate by dividing by the number of episodes.')

args = parser.parse_args()

if args.cmd == 'train':
    match args.trainer:
        case 'base':
            if args.notify < 1:
                args.notify = 1
            if args.batch is None:
                train_base(args.iterations, args.episodes, args.step, args.dim, args.layers, args.learn_rate, args.notify, args.res_dir, args.res_id)
            else:
                train_base_batch(args.batch, args.iterations, args.episodes, args.step, args.dim, args.layers, args.learn_rate, args.notify, args.res_dir, args.res_id)
        case 'cart':
            if len(args.iterations) <= 0:
                args.iterations = parser_training_cart.get_default('iterations')
            for i in args.iterations:
                train_cart(
                    iterations=i,
                    episodes=args.episodes,
                    steps=args.step,
                    latent_dimension=args.lat_dim,
                    latent_task_dimension=args.tsk_dim,
                    layers=args.layers,
                    tasks=args.tasks,
                    learning_rate=args.learn_rate,
                    learning_steps=args.learn_step,
                    discount=args.discount,
                    notify_task=args.notify,
                    fixed=args.fixed,
                    plot_partial=args.plot_partial,
                    double=args.double
                )
        case 'ride':
            if len(args.iterations) <= 0:
                args.iterations = parser_training_cart.get_default('iterations')
            for i in args.iterations:
                train_ride(
                    iterations=i, episodes=args.episodes, steps=args.step,
                    latent_dimension=args.lat_dim, latent_task_dimension=args.tsk_dim, layers=args.layers,
                    learning_rate=args.learn_rate, learning_steps=args.learn_step, discount=args.discount, 
                    notify_task=args.notify, fixed=args.fixed, plot_tasks=args.plot_tasks, plot_partial=args.plot_partial,
                    no_agents=args.agents, grid=SimpleNamespace(l=args.grid[0], w=args.grid[1]), costs=SimpleNamespace(accept=args.costs[0], pick=args.costs[1], move=args.costs[2], miss=args.costs[3], drop=args.costs[4], all_accepted=args.costs[5]),
                    no_tasks=args.task, openness=args.open, learning=True, raw_action=args.raw, double=args.double, decomposed=args.decomposed, clean=args.clean)
        case 'resume':
            if args.extension is None:
                resume(args.source, args.extension, args.plot_partial)
            else:
                for extension in args.extension:
                    args.source = resume(args.source, extension, args.plot_partial)
        case _:
            print(f'No trainer selected! Please select from: {[trainer.metavar for trainer in parser_training._get_subactions()]}')
elif args.cmd == 'plot':
    plot_data(args.source, args.plotter, args.dest, args.episodic)
elif args.cmd == 'run':
    if len(args.sources) == len(args.agents) == 0:
        raise AttributeError('No source or random-agent argument found.')
    for source in args.sources:
        try:
            run_ride(
                src=source, episodes=args.episodes, steps=args.steps,
                grid=SimpleNamespace(l=args.grid[0], w=args.grid[1]), costs=SimpleNamespace(accept=args.costs[0], pick=args.costs[1], move=args.costs[2], miss=args.costs[3], drop=args.costs[4], all_accepted=args.costs[5]),
                no_agents=None, no_tasks=args.task, decomp=args.decomp, clean=args.clean, expert=args.expert, openness=args.open,
                store_trajectories=args.store, fixed=args.fixed, notify=args.notify)
        except AttributeError as e:
            print(e)
    for agents in args.agents:
        if agents > 0:
            run_ride(
                src=None, episodes=args.episodes, steps=args.steps,
                grid=SimpleNamespace(l=args.grid[0], w=args.grid[1]), costs=SimpleNamespace(accept=args.costs[0], pick=args.costs[1], move=args.costs[2], miss=args.costs[3], drop=args.costs[4], all_accepted=args.costs[5]),
                no_agents=agents, no_tasks=args.task, decomp=args.decomp, clean=args.clean, expert=args.expert, openness=args.open,
                store_trajectories=args.store, fixed=args.fixed, notify=args.notify)
else:
    print(f'No command selected! Please select from: {[command.metavar for command in parser_commands._get_subactions()]}')
