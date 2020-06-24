# Corona
## Requirements
You must have Python>=3.8  because I make a lot of use of the walrus operator (:=).

## How to use
Just run `python execute` and it will download everything it needs, execute, and then dump some images into your folder.

The images will be some metrics of the coronavirus, for example it will show per capita deaths (one of the only reliable numbers).

Do not read too much into the recovered/confirmed numbers as those are not reliable for any country, except possibly China, South Korea, and Russia since they were the only countries that took testing seriously, but even those numbers may be off.

## Wishlist
- Refactor. The code is sort of a mess. There are plenty of functions that are useless to most people.
- Make it more customizable.
- Add a GUI

# Possible Bugs
- The country names in the population dataframe is different that the ones in the covid statistics, so there might be an error for some countries. I corrected as many as I could find.
