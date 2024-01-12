"""Translate prediction request for SAX prediction server."""

from typing import Any, Dict, List, Tuple
from saxml.client.python import sax

from saxml.vertex import constants


def user_request_to_lm_generate_request(
    user_request: Dict[str, Any],
    user_request_timeout: int = constants.DEFAULT_PREDICTION_TIMEOUT_SECONDS
) -> List[Tuple[str, Any]]:
  """Translates user request to SAX LM generate request for SAX client.

  Args:
    user_request: the user request for prediction server.
    user_request_timeout: timeout for SAX model server.

  Returns:
    return list of Tuple of (text, sax.ModelOptions).
  """
  if "instances" not in user_request:
    raise ValueError("Request should have key 'instances'")

  instances = user_request["instances"]

  lm_requests = []
  for instance in instances:
    if not isinstance(instance, dict):
      raise ValueError("Each instance should be a dictionary")

    if "text" in instance:
      request_text = instance["text"]
    elif "text_batch" in instance:
      request_text = instance["text_batch"]
    else:
      raise ValueError(
          "Each instance should have a `text` or `text_batch` field."
      )

    option = sax.ModelOptions()
    option.SetTimeout(user_request_timeout)

    # if specified, set ModelOption with extra_inputs
    extra_inputs = instance.get("extra_inputs", None)

    if extra_inputs:
      for key, value in extra_inputs.items():
        option.SetExtraInput(key, value)

    extra_inputs_tensors = instance.get("extra_inputs_tensors", None)

    if extra_inputs_tensors:
      for key, value in extra_inputs_tensors.items():
        option.SetExtraInputTensor(key, value)

    lm_requests.append((request_text, option))

  return lm_requests