# Programming Blog 0x001
Using machine learning to plan wood cutting for paper moulding.

I have recently discovered the Makerspace Bonn near the Cuban embassy here in Bonn. If you are in Bonn and interested in open manufacturing I highly recommend you visit them. They have an entire array of 3D Printers, CNC Routers and they are currently in the process of building a wood and metal shop in their basement.

## The Project
I wanted to get into wood-working for quite some time now and I had the idea of adapting one of my previous adventures into paper-moulding into this. Paper-moulding is very similar to traditionally recycling paper. You chop it up, dissolve it in water and add a bonding agent, like wood glue.
Usually you would then syphon off the paper with a screen, but you can only get flat shapes with that process. Those sheets are then folded into shapes like egg-cartons for example, but if you need something a little bit stronger you can also press the water out with a mould.
This way you can get cylinder-type shapes with shaped tops and bottoms and the material feels very similar to a light wood. I don't want to make a completely new mould for every shape I make so I decided to make a standard box-shaped mould and then only 3D-Print fitted shapers for the top and the bottom to form the cylinders.

The box is made out of two different type of parts: four wall pieces which are glued together and two top/bottom pieces which are bolted on to provide the necessary pressure to remove the water, essentially creating a make-shift vice.

[Picture here]

If I want to make just one mould it's a pretty straight forward process: The local hardware-store sells two sizes of pretty cheap pine-wood with the right thickness:
- 600x200x18mm and
- 600x400x18mm

But I can fit all pieces on the smaller size, with the wall being 100x120x18mm and the top and bottom pieces being 140x140x140mm. This however is a pretty wasteful arrangement with only about 73% of the wood being used and if you want to make multiple of these moulds you can save quite a bit of money choosing a more efficient design. Over all I found 5 different arrangement of parts on the boards.

[Pictures of all 5 designs here]

Remember: We need 4 sides and 2 end-parts for each mould, but if you count the parts on the boards you can clearly see, that only option A has the perfect ratio of end-to-side parts. Option B has too many sides, C has no ends at all and the options D and E have more end-parts. We now need to find a good combination of these different cutting patterns to fit our part-requirements as closely as possible.

## Making my CPU do the thinking for me

Now, I am not a very smart person. I am also not very good at crunching big amounts of data and I am really bad at exploring a solution-space, so I am getting my computer to do it. Having the computer look at every single possible combination of boards would be pretty slow (especially on my I7), so I make him look at one, look at similar ones and if my little box of thinking rock finds a better solution than it currently holds it takes that as its next starting point and repeats the process there.

This is called a "Genetic Algorithm" because I essentially make "mutations" to "offspring" and then apply evolutionary pressure by only having the best of each batch "reproduce". Mind you, I am not fitting a model to a dataset here, like you would do with classification models for example, but I am optimizing for well defined rules.

### Defining the problem

First we need to set the boundaries. We need a way to evaluate the different solutions to the problem. For this I first transcribed the different options into a format my computer could understand.

```python
#File: wood.py
options = {
    # Name: [Sides, Ends, Price, Waste in cm²/16]
    "A": [4, 2, 4.99, 82],
    "B": [11, 4, 9.99, 74],
    "C": [20, 0, 9.99, 0],
    "D": [8, 5, 9.99, 66],
    "E": [2, 3, 4.99, 117],
    "-": [0, 0, 0, 0]
}
```

Each option saves the number of sides and end pieces on the board, as well as the price of the boards and the amount of unused material. That last number is scaled down as a result of how I calculated those numbers, but they are linearly proportional to the amount in square centimetres.

### Good? What does that even mean?
Next we need to evaluate the combination of options chosen. For this we first need to know how many parts a given combination produces. Since that's in our dataset, we can just add up the values:

```python
def get_parts(combination):
    """
    Returns tuple of parts from combination
    """

    total_sides = 0
    total_ends = 0

    for option in combination:
        total_sides += options[option][0]
        total_ends += options[option][1]

    return total_sides, total_ends
```

From this we can now get how many moulds you can produce from the combination.
Each mould needs four sides and two end pieces so we can just divide them as integers and return the smaller one:

```python
def get_moulds(combination):
    """
    Takes a combination and calculates number of moulds that can be made from that combination of cutting patterns.
    """

    sides, ends = get_parts(combination)
    moulds_s = sides // 4
    moulds_e = ends // 2

    #Return the smaller number
    return moulds_s * (moulds_s < moulds_e) + moulds_e * (moulds_s >= moulds_e)
```

The return statement may look a bit wonky to you but it's just a branchless way of writing:

```python
  if moulds_s < moulds_e:
    return moulds_s
  elif  moulds_s >= moulds_e:
    return moulds_e
```

A big appeal of branchless programming, especially in compiled languages is, that it runs faster in some instances, because the processor can actually prepare the next statements while the previous one is still executing. I have no idea if this piece of code is faster, I would doubt it actually.
I did it, because it's easier for me to read. Instead of thinking "If this is smaller than that then..." I just think "Return the smaller number". However documentation is really important here, because if there is no documentation and you don't already know what is supposed to happen here, this style of programming is often times a lot harder to understand.

Now that we have these to predicates we can actually calculate the loss-function of our machine-learning algorithm.

To calculate the wasted material I get the number of moulds that can be made from these combinations and if that number is smaller than the number of moulds required, I fill the difference with pattern A, since it produces exactly one mould. Then I add up the waste of these options as well as the area of the parts I would produce too many off.

```python
def evaluate(required_moulds, combination):
    global options
    """
    Takes a number of moulds to produce and a list of combinations [\"A\", \"E\", \"D\",\"E\",...] and returns a score which is to be minimized.
    """

    # ==Minimize waste==
    waste = 0

    # Pad with A-Pattern
    moulds = get_moulds(combination)
    if required_moulds-moulds > 0:
        combination = combination + ["A"]*(required_moulds-moulds)

    #Cutting Waste
    for pattern in combination:
        waste += options[pattern][3]

    # Add Wasted Parts
    perfect_sides, perfect_ends = moulds*4, moulds*2
    total_sides, total_ends = get_parts(combination)

    waste += (total_sides-perfect_sides)*30+(total_ends-perfect_ends)*49

    return waste
```

Let's address the elephant in the room: Why am I not optimizing for cost?
The answer here has multiple facets. First off all, the cheapest option is to not do it at all, so the computer would tell me just not to buy any wood. But I could solve that in a similar fashion as I have solved the problem of underproduction of parts. Just pad with option A.
Second of all: I'm not working in my shop. I don't want a lot of small parts flying around, because I will have to take them with me.
Third of all, there's a fixed conversion rate between cost and area, so it doesn't really matter. The big piece is twice the size and costs twice as much.

### I attended a ninth grade biology class
Now we need to do the whole evolution thing. For this we first need to "mutate" the current combination. To do this I select a random slot of a copy of the combination and assign it to a random pattern.

```python
# File: evolution.py
from copy import copy
from random import randint as random
import wood

def mutate(combination):
    next_combination = copy(combination)
    pos = random(0, len(combination)-1)
    new = wood.names[random(0, len(wood.names)-1)]

    next_combination[pos] = new
    return next_combination
```

I do this for as many "offspring" I want. The more offspring I generate the more diverse is the "genetic pool", so I can more accurately steer towards the minimum but it will also take more time to cycle through one generation. I decided on 1/3 the length of the array and that seems to work well enough.

Next I need a function that generates and evaluates all thise offspring and chooses the best one:

```python
def get_next_combination(combination, n_children, molds):

    # Generate Mutations
    children = list()
    for i in range(n_children):
        children.append(mutate(combination))

    # Find best children
    best_child = combination
    best_score = wood.evaluate(molds, combination)

    for child in children:
        score = wood.evaluate(molds, child)
        if score < best_score:
            best_score = score
            best_child = child

    return best_child, best_score
```

As a basis I chose the current score. If no child can improve on the current combination, the computer will choose the old one and try again. (Experienced AI-engineers, calm down. We'll talk about this later.)

Now we just need to run it:
```python
from sys import argv

def main():
    moulds = int(input("Moulds to be made: "))
    max_generations = int(input("Maximum number of generations: "))

    perfect = wood.get_perfect_amounts(moulds)
    length = moulds

    if len(argv) == 1: #This lets me tweak the number of children
        children_per_gen = length//3+1
    else:
        children_per_gen = int(argv[1])


    combination = ["-"]*length
    #combination = ["A"]*moulds

    print("Initial Score:", wood.evaluate(moulds, combination))

    i = 0
    scores = list()
    x = list()
    while i <= max_generations:
        try:
            combination, score = get_next_combination(combination, children_per_gen, moulds)
            #print(score, end=", ")
            i += 1

            scores.append(score)
            x.append(i)

            if score == 0:
                break
        except KeyboardInterrupt:
            #This allows me to break out and still get the current results, if it takes too long.
            break

    show_results(combination, score, perfect, moulds, scores, x)

```

The length of the array that represents the combination is equal to the number of moulds requested because we know that there is a solution within these constraints and either it is the best solution, or the best solution will use less material.

## SO..... How did I do?
Short answer: Pretty ok.
Long answer: I works fine, but there are some improvements to be made.

[Picture of 50-30]
Here is an example where I asked to calculate for 50 moulds with a maximum of 30 Generations.
You can see, how the waste-score starts high, at 15000, but quickly drops in a linear fashion. It then plateaus at 1589 points.

```
Moulds to be made: 50
Maximum number of generations: 30
Initial Score: 15000
Missing 0 moulds. 50/50

Score: 1589
A*1
B*5
D*15
E*1


Difference:
Sides: 1
Ends: 0

Price: 219.77€
```

[Picture of 200-100]
Requesting larger amounts (200) looks similar. Here something very interesting happened, and it turned out to be reproducable:

```
Moulds to be made: 200
Maximum number of generations: 100
Initial Score: 60000
Missing -4 moulds. 204/200

Score: 6072
A*0
B*66
C*0
D*18
E*0


Difference:
Sides: 16
Ends: 8

Price: 839.16
```
The most efficient solution the computer could find involves enough waste to cut multiple other moulds from.

### Problems

Remember when I told the experienced engineers to calm down?

> If no child can improve on the current combination, the computer will choose the old one and try again.

This sounds very sensible at first. We want the next generation to be better than the last. Imagine there is a ball are on a hill. The ball will naturally tend downwards and will roll down the side of the hill.

```
_
 \o <- Ball
  \    __
   \__/  \
          \_____
```

In this scenario it will hit the first local minimum, but if it can't clear the second bump it will never reach the global minimum. In more specific terms here that means that multiple changes to the combination may be required to make an improvement. During testing I actually discovered this to be a problem. Requesting large number of moulds twice would sometimes produce two different accuracies, which means that at least one instance is stuck in a local minimum.

Sometimes you need to be willing to take steps backwards to jump further ahead, just how that ball would need to move back up that hill to reach the lowest point.

Another option would be to mutate multiple parts of the "genome" at once, increasing it's volatility and this maximizing the chance of the figurative ball to jump over that hill at random.

However overall the algorithm can produce 10 to 20 percent more cost-efficient solutions than cutting all parts from the smallest piece of wood. The managers of a machine shop would kill for this kind of improvement.

It was a fun project and I'm glad I've done it. I might release the full source code on GitHub one day.
