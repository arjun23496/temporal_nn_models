### Approach 1

Sims is a possible game that we can use as an environment but I tried searching for ways in which we can directly 
interact with Sims using python scripts and it is not pretty. There seems to be lot of nuances for scripting in sims and 
since we are time constrained, it might not be a great idea. The other option is as we discussed before, MineRL which 
is based on Minecraft game. This game has suffifcient community support for scripting. With that said, the actions of 
different will have to be different to accommodate the nature of the game.

I found this paper MineRL: A Large-Scale Dataset of Minecraft Demonstrations (https://arxiv.org/pdf/1907.13440.pdf) 
that uses a dataset consisting of many hours of *human* gameplay as a dataset for training RL agents. The high level 
actions they have are:

- navigate
- tree chop
- Obtain item (bed, meat, pickaxe, diamond)

To put this into a framework that will work for us, we will have to club actions together according to the timeline 
I described in the slides.

For example: the time based actions are:
- obtain meat (during meal times) maybe chop wood (during meal times), obtain bed(at night)

The event driven actions are:
- navigate (to chop wood, obtain things like meat, pickaxe)

The sporadic actions are:
- chop wood, obtain diamonds

I am not familiar with the game to add in any extra actions but building a timeline with these actions seems like 
something we will be interested in. So for the RL problem we define,

- State: all the state information provided by the MineRL framework+time so that an agent can learn the time based 
- actions. Alternatively, time can be removed from the state with the burden of timekeeping falling on our network 
- encoding the policy.
- Actions: any action allowed by the game
- Reward - This will change according to what high level action is currently being performed.
- Policy - this is what will have to encoded by our model. Time based actions should happen periodically so time is an 
- important feature the policy will have to leverage.

### Approach 2

The other approach which can be made directly from the Sims
dataset will be a forecasting problem (I am getting the hint that you
might be meaning this). I build a timeline of events of a sim performing
certain actions and the objective of the model is to build a
distribution over the possible future actions the Sim is likely to
perform. This approach has a dataset (the video recordings of actions
being performed). The model will possibly do two things: recognize the
action from the current frame and then extrapolate the possible set of
actions in the future frames (or time).
