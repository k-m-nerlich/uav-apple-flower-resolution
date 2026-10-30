# Annotation protocol (English translation)

Translation of `annotation/ANNOTATIONSPROTOKOLL.md`, the written protocol given
to the second annotator (B). B had no prior experience with orchards or aerial
images and received no instruction beyond this protocol.

## Task

You receive 36 image sections from an apple orchard, taken by a drone from about
5 m. Each section shows about 1.7 × 1.7 m of ground with part of a tree row.

**Set a point on every apple flower you recognise.**

## Rules

1. **One point per single flower**, placed in the centre of the flower. Not one
   point per cluster: if you see five flowers close together, set five points.
2. **Cut off at the image border**: count if more than half of the flower is visible.
3. **In shade**: count if you recognise it as a flower.
4. **Buds and withered flowers** do not count. Only open white flowers.
5. **Doubtful cases**: use the label `unsicher` (uncertain) instead of guessing.
   These points are evaluated separately; their number is a result, not waste.
6. **Empty tiles exist.** If nothing is visible, save without any point.

## Important for the evaluation

- **Do not search longer than a first look needs.** Set yourself a time limit of
  about 2 to 3 minutes per tile and keep it.
- **Do not look at other annotations**, neither before nor during the task.
- **Do not go back and revise earlier tiles** after learning more. Work through once.
- **Keep the order** in which the tiles are shown.

Zooming is allowed and useful — the flowers are small.

## Technical procedure

Label Studio with the labelling configuration in `annotation/labelconfig.xml`:
key `1` selects `Bluetencluster` (this is the single flower; the name is
historical), key `2` selects `unsicher`. Export as JSON.
