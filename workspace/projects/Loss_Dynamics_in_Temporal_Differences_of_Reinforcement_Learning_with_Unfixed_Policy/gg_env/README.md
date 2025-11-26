# CS182 Individual Programming Assignment 4: Goblets and Ghouls

This assignment walks you through implementing tabular Q-Learning for the same grid-based reinforcement learning world. Your agent will learn a policy by interacting with a stochastic environment, gathering rewards (goblets) and avoiding hazards (obstacles + moving ghosts).

## Game

In the individual and group assignments, you will be controlling an adventurer (agent) to collect goblets in a discrete gridworld. However, because the world you're exploring is booby-trapped your agent doesn't always take your specified action. Instead, your transition function is stochastic and takes a particular action with some probability given your desired action. This transition function probability distribution is represented as a list of 4 probabilities. The list can can be interpreted as follows:

Actions are ordered as $(\text{UP}, \text{RIGHT}, \text{DOWN}, \text{LEFT})$ and the transition parameters are $[a,b,c,d]$ with $a,b,c,d\ge 0$ and $a+b+c+d=1$.

-   $a$: take the **intended** action
-   $b$: slip **one step clockwise**
-   $c$: slip **two steps** (opposite direction)
-   $d$: slip **one step counterclockwise**

<div align=center>

| intended \ actual |    UP | RIGHT |  DOWN |  LEFT |
| ----------------- | ----: | ----: | ----: | ----: |
| **UP**            | \(a\) | \(b\) | \(c\) | \(d\) |
| **RIGHT**         | \(d\) | \(a\) | \(b\) | \(c\) |
| **DOWN**          | \(c\) | \(d\) | \(a\) | \(b\) |
| **LEFT**          | \(b\) | \(c\) | \(d\) | \(a\) |

</div>

In words: the first entry \(a\) is the probability of executing the intended action; the \(i\)-th subsequent entry gives the probability of slipping \(i\) steps **clockwise** from the intended action.

The adventurer takes the first goblet that they find. Some of the goblets in the maze are made of fool's gold and provide negative reward to the adventurer. The adventurer immediately receives the reward (or penalty if the goblet is made of fool's gold) associated with a goblet when they transition into a cell that contains a goblet.

In this assignment, however, there is a ghoul that is attempting to prevent your adventurer from collecting the goblets. The ghoul is able to pass through walls.

## Q-Learning

In Q-Learning (off-policy TD control), we learn the action-value function Q(s, a) directly from experience by repeatedly:

-   choosing actions via an exploration strategy (epsilon-greedy),
-   stepping the environment to observe next state and reward,
-   applying a temporal-difference update toward the target R*{t+1} + γ max_a' Q(S*{t+1}, a').

For this assignment you will complete the core training loop and policy extraction in `src/gg/q_learning.py`:

1. `q_learning_step()` - Perform one transition and apply the Q-learning update
2. `extract_policy_from_q()` - Build a greedy policy from the learned Q-table
3. `train()` - Orchestrate episodes of interaction until training completes

Q-Learning updates an estimate of the optimal action-value function using the rule:

```
Q(S_t, A_t) ← (1 − α) · Q(S_t, A_t) + α · [ R_{t+1} + γ · max_a Q(S_{t+1}, a) ]
```

-   α is the learning rate
-   γ is the discount factor
-   If the next state is terminal, the bootstrapped term is omitted (target becomes just R\_{t+1}).

In this environment, actions are stochastic: you select an intended action, but the world may execute a different one according to transition probabilities in `config.agent.transition`, ordered as `[intended, right, back, left]` relative to your intended direction.

## Part 1: Implement `q_learning_step()`

Performs a single data-collection + TD update step given the current state and Q-table.

High-level pseudocode:

```
1. Choose intended action index A_t via epsilon-greedy on Q(S_t, ·)
2. Sample actual executed action index using the stochastic model
3. Step the environment
4. Compute TD target:
   - If next_state.done: target = reward_next
   - Else: target = reward_next + γ · max_a Q(next_state, a)

5. Apply tabular Q update at (state, intended_idx)
   - Q(S_t, A_t) ← (1 − α) Q(S_t, A_t) + α · target
   - Use apply_q_update(q, state, intended_idx, alpha, target)

6. Return next_state
```

## Hint: What should your state representation be with the ghost in mind?

## Part 2: Implement `extract_policy_from_q()`

Build a greedy, deterministic policy that maps each agent cell to its best action under the learned Q-values. Because Q depends on both agent and ghost positions, we aggregate over ghost positions when deciding the best action for each agent cell.

High-level pseudocode:

```
1. Create policy array π with shape (width, height) and dtype=object

2. For each agent cell (x, y):
   a. Extract all Q-values over ghost positions: q_slice = q[x, y, :, :, :]  # shape (W, H, |A|)
   b. Compute expectation over ghost locations: expected = mean(q_slice, axes=(0,1))  # (|A|,)
   c. best_idx = argmax(expected)
   d. π[x, y] = ACTIONS[best_idx]  # store an Action object

3. Return π
```

Notes:

-   Store `Action` objects in π, not indices.
-   Use the action order `[Up, Right, Down, Left]` consistently.

---

## Part 3: Implement `train()`

Runs multiple episodes of interaction, applying `q_learning_step` repeatedly and finally extracting the greedy policy.

High-level pseudocode:

```
1. Initialize Q-table.
2. Initialize RNG: seed from config.generation_seed

3. For each episode:
   a. Start from initial state
   b. For t in [1, number_of_episodes]
      - For a trajctory loop, compute a Q-learning step. If reached a terminal state, break and try another trajectory.

4. Return extracted policy from Q-table.
```

Hints: Try playing around with how many episodes you'll need to beat the ghost and get the goblet!

## Installation

### `uv`

You should have `uv` installed from the previous programming assignment. If so, you should feel free to skip to [Installing Project Dependencies](#installing-project-dependencies). Otherwise, we've duplicated the instructions for installing `uv` below. We highly recommend that you use `uv` to manage your virtual environments for this course. Feel free to post on Ed or stop by office hours if you have any questions or trouble setting up your environment!

#### Installing `uv`

A headache-free way to start running the starter code is via the [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager. To install (on Mac, Linux, WSL):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> [!NOTE]
> The above command requires `curl` installed! You'll most likely have this installed, already but if not it should be a small `brew/apt/pacman install` away. Alternatively, if you have `wget` already installed you can use
>
> ```
> wget -qO- https://astral.sh/uv/install.sh | sh
> ```

Alternative approaches exist to install `uv` which you are invited to explore in the [Astral documentation](https://docs.astral.sh/uv/getting-started/installation/).

#### Installing Project Dependencies

One of the major superpowers of `uv` is its ability to manage different versions of Python using a combination of virtual environments and configuration files. Initialize this project's virtual environment and install all the required dependencies by running the following command **in the root of your cloned GitHub repository**:

```
uv sync
```

> [!NOTE]
> You'll need to create a new virtual environment for each assignment!

This should install an appropriate Python version (3.13) and all the project's dependencies including `pytest`.

### Selecting your Python Interpreter

Using typed Python can highly improve your programming (and especially your debugging experience). To enable type checking in VSCode:

-   Open Settings (`CTRL`/`CMD` + `,`)
-   Search for `Type Checking Mode`
-   Set the mode to your desired level of strictness (`standard` is a good default!)

<div align="center">
    <img src="docs/type_checking.png" width="1000"/>
</div>

To ensure that all the type stubs are resolved for your installed dependencies, you have to select the correct Python interpreter. To do so:

-   Open the Command Pallette (`CTRL`/`CMD` + `SHIFT` + `P`)
-   Type and select `Python: Select Interpreter`p
-   Choose the virtual environment located in the stencil directory.

<div align="center">
    <img src="docs/interpreter_selection.png" width="1000"/>
</div>

For this to be correctly detected, be sure that the opened folder in VSCode is your cloned Github repository!

#### Running your Agent

To launch and interact with any of the Maze agents, use:

```
uv run gg [-h] [--config CONFIG] (--render-policy RENDER_POLICY | --train-policy TRAIN_POLICY)
```

```bash
# Train your value iteration policy
uv run gg --train-policy /path/to/policy/output

# Render your value iteration policy
uv run gg --render-policy /path/to/policy/
```

To run the provided tests (or ones you created) use:

```
uv run pytest
```

> [!NOTE]
> No sourcing required! If you prefer the virtual environment folder to not be called `.venv` then you can pass a name to the `uv venv` command. However, this would require setting the `UV_PROJECT_ENVIRONMENT` variable to this name either prior to every `uv sync` and `uv run` command, exporting the variable in every terminal session, or using some external dependency to manage your environment variables like [`direnv`](https://direnv.net/).

## Tips and Debugging

We use `pytest` to write and run tests for your assignments. `pytest` comes with a ton of handy and useful flags to help you test and debug parts of your code! Here's a list of useful flags to pass to `pytest` when debugging your assignment.

### Selecting Tests

```
pytest                      # run all tests (auto-discovery)
pytest path/                # run tests under a directory
pytest tests/test_x.py      # run tests in a file
```

### Failure Control

```
pytest -x                  # stop after first failure
pytest --maxfail=3         # stop after 3 failures
pytest --lf                # run last-failed tests only
pytest --ff                # run failures first, then the rest
pytest --sw                # stepwise: stop on first fail, resume next time
```

### Output

```
pytest -q                  # quiet (less output)
pytest -v                  # verbose (test names)
pytest -vv                 # extra verbose
pytest -s                  # don't capture stdout/stderr (show prints)
pytest -rA                 # show summary for all outcomes (reasons)
pytest -l                  # show local variables in tracebacks
pytest --tb=short          # shorter tracebacks
pytest --tb=line           # one-line tracebacks
pytest --durations=10      # show 10 slowest tests
```

### Common Combinations

```
# Fail fast with clear reasons
pytest -x -vv -rA --tb=short

# Re-run only what failed last time, show prints
pytest --lf -s -vv

# Redirecting output to a file for readability
pytest -s > foo.txt
```
