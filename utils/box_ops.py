import tensorflow as tf
from tensorpack.tfutils.scope_utils import under_name_scope


@under_name_scope()
def area(boxes):
    """
    Args:
      boxes: nx4 floatbox

    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

@under_name_scope()
def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

# @under_name_scope()
# def tf_clip_boxes(boxes, shape):
#   """
#     boxes: n x 4
#   """
#   boxes = tf.reshape(boxes, [-1, 4])
#   x1y1 = tf.slice(boxes, [0, 0], [-1, 2])
#   x2 = tf.slice(boxes, [0, 2], [-1, 1])
#   y2 = tf.slice(boxes, [0, 3], [-1, 1])
#   x1y1 = tf.clip_by_value(x1y1, tf.constant(0.0), tf.constant(10000.0))
#   x2 = tf.clip_by_value(x2, tf.constant(0.0), tf.to_float(shape[1]))
#   y2 = tf.clip_by_value(y2, tf.constant(0.0), tf.to_float(shape[0]))
#   x1 = tf.slice(x1y1, [0, 0], [-1, 1])
#   y1 = tf.slice(x1y1, [0, 1], [-1, 1])
#   return tf.stack([x1, y1, x2, y2])

@under_name_scope()
def tf_clip_boxes(boxes, shape):
  """
    boxes: n x 4
  """
  boxes = tf.reshape(boxes, [-1, 4])
  x1 = tf.slice(boxes, [0, 0], [-1, 1])
  y1 = tf.slice(boxes, [0, 1], [-1, 1])
  x2 = tf.slice(boxes, [0, 2], [-1, 1])
  y2 = tf.slice(boxes, [0, 3], [-1, 1])

  x1 = tf.clip_by_value(x1, tf.constant(0.0), tf.to_float(shape[1]))
  y1 = tf.clip_by_value(y1, tf.constant(0.0), tf.to_float(shape[0]))
  x2 = tf.clip_by_value(x2, tf.constant(0.0), tf.to_float(shape[1]))
  y2 = tf.clip_by_value(y2, tf.constant(0.0), tf.to_float(shape[0]))

  x1 = tf.squeeze(x1)
  y1 = tf.squeeze(y1)
  x2 = tf.squeeze(x2)
  y2 = tf.squeeze(y2)

  return tf.reshape(tf.transpose(tf.stack([x1, y1, x2, y2])), [-1, 4])