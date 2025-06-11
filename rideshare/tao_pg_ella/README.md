# Open-Task Policy Gradient Lifelong Learning Algorithm in the OASYS Rideshare Domain

This branch features an adaptation to the policy gradient lifelong learning algorithm (PG-ELLA), described in http://proceedings.mlr.press/v32/ammar14.pdf, that has been extended for use in open-task environments, dubbed open-task PG-ELLA (OTPG-ELLA). The branch should match rideshare-tao except for this readme and the key files to the featured implementation, all prefixed with "ellapig". Said prefix is just a reference to PG-ELLA, written as a word with PG moved to the end to avoid frequently writing the slower all-caps name in hand-written notes.

## How to Use

For the most up-to-date rideshare-specific training, see Ellapig Trainer. It is well documented for integration into a new project and example calls are presented at the bottom of the document for immediate training and evaluation.

For other training and older, more feature-rich rideshare training, try the console command `python ellapig.py -h`. Rideshare training through this method is deprecated and may not work as desired.

For integration into your own work or further development, see the description section on Ellapig Net or the documentation provided on the file of the same name.

## Description

### Ellapig Net

This is the main file containing OTPG-ELLA. This includes the actor-critic base learner, two implementations of OTPG-ELLA and their dependencies, and a compact policy-only class. Additionally, this is where PyTorch checks device type for GPU utilization.

#### Base Learner

This is defined at the bottom of the file, using the classes ACnet and ACagent. The former is an wrapper for PolicyNetwork, with an initializer and some helper functions specific to the actor-critic implementation. The latter, ACagent, conatins two Acnets: the actor and the critic, as well as loss calculation functions and an optimizer over both the critic and actor networks. Furthermore, ACagent has a function for calculating the Hessian gradient of the actor network.

The Hessian function used to cover both actor and critic, but due to some errors found resulting from the change to PolicyNetwork-defined policies (rather than a simple tensor) after having already moved away from Duellapig (see below), the critic was unecessary and not implemented. Implementation should be  a relatively simple affair, just duplicate the function using self.critic.params() to define test_net, and wrap both that and the actor calculations for the full set. This should work, but has not been tested.

#### Linellapig

This is the main and functional implementation of OTPG-ELLA. It is intialized from a variety of policy and learning parameters, and it contains functions for forwarding observations through the learned policies to return an action over the joint action space of observed tasks and for learning and optimizing the shared and task-specific latent spaces, including filtering to convert general trajectories into task-specific ones and base-learner policy optimization over those tasks. Observations are input as a dictionary of fixed-size task-specific float vectors. Trajectories are lists of episodes which themselves are lists of steps that each contain the set of observations, th chosen action, and the returned rewards for that step. Note that while the type hints say steps are expected as a tuple, they actually expect named attributes of observation, action, and reward. E.g. \[step.observation\[task\], step.action, step.reward for episode in trajectories for step in episode for task in step\].

#### Duellapig

This is a deprecated implementation of OTPG-ELLA, that was designed much like the above, but to maintain two sets of shared and task-specific latent spaces for the actor and critic respectively. It requires a lot more memory to account for that, and in practice tended to crash from memory problems. As such, functionality in its current state is not guaranteed and in fact it can be expected not to function, due in part to the limitations of the currently-in-use Hessian function described in the base learner section above. That said, it remains in the file as a potentially useful avenue for future work, should the inter-task relativity of Linellapig's policies prove insufficient. If developed, it would be good to find a way to estimate the local optimal and Hessian gradients to save on storage space, but at present, such attempts have greatly extended the time required to run the algorithm, to the poin of being beyond reasonable use.

#### Simpellapig

This is not an inplementation of OTPG-ELLA, but does allow for much more compactly-saved policies from the above OTPG-ELLA implementations. Given a learned policy, this class stores only the policy dimensions and the learned actor shared and task-specific latent spaces. It contains one function, used for forwarding environmental observations to return an action.

### Ellapig Data

This contains various plotting and loading/storing functions for the netwroks described in Ellapig Net. The most up-to-date of these functions are the plotters for training, rewards, policies, and pooling, and they are designed to work with the newer rideshare trainer.

### Ellapig (.py)

This is a deprecated console tool that has calls for functions in Ellapig Training and Ellapig Data. The data calls should still be fully supported, as should the base-learner and cartpole training calls, but the rideshare training may have problems with a since-updated domain.

### Ellapig Training

This contains the old OTPG-ELLA training functions. The base and cart trainers should still work as designed, but the rideshare training functions may not work with the more up-to-date versions of the rideshare environment. They are retained because they included some extra functionality not present in the newer rideshare trainer, such as fixed training and expert-trajectory training.

#### Train Base & Train Base Batch

These functions are used to test base-learner functionality in the CartPole environment. The latter generates training over several sessions and averages the final results.

#### Train Cart

This function trains OTPG-ELLA in the CartPole environment. It includes a way to test multiple vs a single task by representing different episodes as different tasks, but this is not true task openness.

#### Train Ride

The old rideshare trainer, see ellapig.py help command for more detail on what the various arguments do.

#### Make Clean Trajectory

Function for initializing expert trajectories. Agents will start at their longest assigned passenger, then accept all passengers assigned to them, then pick them up in order, then drop passengers off in reverse order such that all passengers assigned to that agent are optimally pooled. Notably, trajectories generated from this ignore other agents, so practical competitiion may be a later problem, but the focus was giving agents examples where pooling worked and worked well. This is just the initializer and task assigner; actual policies are run in Pool Targets.

#### Pool Targets & Pool Targets Decomposed

The policies used in expert trajectories, see the Make Clean Trajectories Section above.

#### In-Path & Break-Path Pooling

These functions generate passengers who share parts of their routes, allowing optimal pooling thereover. In-path trajectories are all simple right-angle paths, while break-path pooling can include multiple turns.

#### g Mode

This runs a single, specific expert trajectory created by g, it was dropped fairly quickly and does not produce good results.

#### Resume

Resumes previous OTPG-ELLA training in the CartPole or Rideshare environments. Requires agent and configuration data, and ideally prior learning statisitcs for best use.

#### Run Ride

Evaluates a policy for the rideshare environment learned by OTPG-ELLA (deprecated).

### Ellapig Trainer

This is the new rideshare trainer, and it is well documented, so look there for more detail, but generally this file has various data classes and a main class for conductiong OTPG-ELLA training in the rideshare environment.

Most of the data classes are used for sub/pre-initialization in main class functions, but the trajectory class is a better descriptor than the tuple from Ellapig Net. It also includes a per-step rideshare-specific reaction to account for potential differences from chosen and actual actions, as well as a tasks variable used in recording statistics.

#### Environment Customization

This section includes functions used to initialize the rideshare environment according to user specifications or an exisitng configuration file.

#### Environment Helpers

This section contains in-between functions for converting the numpy/graph-based rideshare IO into OTPG-ELLA IO. Specifically, the interpreter converts the observation from a variably-sized and multidimensional numpy array into a dictionary of fixed-size single-dimensional float tensors for use by OTPG-ELLA, and the exterpreter converts an integer output from OTPG-ELLA into a graph action node (using the converted observation) for rideshare.

#### Agent Customization

This section has various functions for initializing the trainer's agents based on user specs or existing files containing pickled agents. the New & Load functions are for creating individual agents, while the rest (re)initialize the entire set of agents.

#### Experiment Customization

Despite its name, this section not only contains functions for setting up experiments, but also running and evaluating them.

Run Episode and Gather Trajectories are purely for running the rideshare environment with the trainer's current setup, but they need more specific instructions on how it should be run, which is where the Train and Run (and their std variants) come in. Both expect the user to have initialized an agent-environment setup before being called. Train is used for OTPG-ELLA learning to create new or optimize existing policies, and data gathered reflects that experience. Run is used for policy evaluation and does not perform any learning, with gathered data only reflecting the trajectories generated. Run also includes an option for processing trajectory datain a manner separate from the origial design through a user-definied function in the keyword-only process argument. That different processing was used to create the various evaluation functions, all prefixed with eval, these functions save to CSVs rather than maintaining data in memory. The stored data is used to create various plots relfecting their namesakes. E.g. Eval Training provides reward scores from an initial random policy through various check points made in training to show learning progress. The corresponding plotting functions found in Ellapig Data were made with specific use cases in mind, and they may need some tweaking to fit newer user needs. Alternatively those calls are just in the last line of the evaluation functions, so they can easily be replaced. Furthermore, the CSV data storage allows users to do whatever with the gathered data after the fact, without changing the functions themselves.

Run Episode also has two lines commented out for episodic fixed-location driver initialization. This was only used in last minute in evaluation, and was the same for all of it, so the functionality lacks full integration. That said, a quick preview of some of what would be needed is in place.

#### Example Calls

Not really a formal section, the bottom of this file contains examples of training and evaluation function calls that can be put in a new tool or uncommented/written-anew for direct usage. This file does not really have console support, so either run python in console and use these calls after importing the file or prep a file with these calls to use the trainer.
