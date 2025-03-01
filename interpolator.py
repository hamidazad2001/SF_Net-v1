# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import feature_extractor
import fusion
import options
import pyramid_flow_estimator
import util
import tensorflow as tf

import gin.tf
def _relu(x: tf.Tensor) -> tf.Tensor:
  return tf.nn.leaky_relu(x, alpha=0.2)

def create_azadegan_model(x0: tf.Tensor, x1: tf.Tensor, config: options.Options) -> tf.keras.Model:
  
  image1 = tf.expand_dims(x0, 1)
  image2 = tf.expand_dims(x1, 1)
  input_img = tf.concat([image1, image2], axis=1)
  
  conv01 = tf.keras.layers.Conv3D(32, kernel_size=(2, 3, 3), activation = _relu, padding = 'same')(input_img)
  conv001 = tf.keras.layers.Conv3D(32, kernel_size=(2, 3, 3), activation = _relu, padding = 'same')(conv01)
  conv0001 = tf.keras.layers.Conv3D(32, kernel_size=(2, 3, 3), activation = _relu, padding = 'same')(conv001)
  conc01 = tf.concat([conv01, conv001, conv0001], axis=-1)
  conv02 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc01)
  down01 = tf.keras.layers.AveragePooling3D((1, 2, 2), padding='same', data_format='channels_last')(conv02)
  conv03 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(down01)
  down02 = tf.keras.layers.AveragePooling3D((1, 2, 2), padding='same', data_format='channels_last')(conv03)
  conv04 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(down02)
  down03 = tf.keras.layers.AveragePooling3D((1, 2, 2), padding='same', data_format='channels_last')(conv04)
  conv05 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(down03)
  down04 = tf.keras.layers.AveragePooling3D((1, 2, 2), padding='same', data_format='channels_last')(conv05)
  conv06 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(down04)
  down05 = tf.keras.layers.AveragePooling3D((1, 2, 2), padding='same', data_format='channels_last')(conv06)
  conv07 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(down05)
  down06 = tf.keras.layers.AveragePooling3D((1, 2, 2), padding='same', data_format='channels_last')(conv07)
  conv08 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same')(down06)
  upd01 = tf.keras.layers.UpSampling3D((1, 2, 2))(conv08)
  conc02 = tf.concat([upd01, conv07], axis=-1)
  conv09 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc02)
  upd02 = tf.keras.layers.UpSampling3D((1, 2, 2))(conv09)
  conc03 = tf.concat([upd02, conv06], axis=-1)
  conv10 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc03)
  upd03 = tf.keras.layers.UpSampling3D((1, 2, 2))(conv10)
  conc04 = tf.concat([upd03, conv05], axis=-1)
  conv11 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc04)
  upd04 = tf.keras.layers.UpSampling3D((1, 2, 2))(conv11)
  conc05 = tf.concat([upd04, conv04], axis=-1)
  conv12 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc05)
  upd05 = tf.keras.layers.UpSampling3D((1, 2, 2))(conv12)
  conc06 = tf.concat([upd05, conv03], axis=-1)
  conv13 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc06)
  upd06 = tf.keras.layers.UpSampling3D((1, 2, 2))(conv13)
  conc07 = tf.concat([upd06, conv02], axis=-1)
  conv14 = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (4,1,1))(conc07)
  conv15 = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation = _relu, padding = 'same', strides = (2,1,1))(conv14)

  squeeeexe = tf.squeeze(conv15, axis=[1])
  output = tf.keras.layers.Conv2D(3, kernel_size=(1), padding = 'same')(squeeeexe)
  output = tf.clip_by_value(output, clip_value_min=0, clip_value_max=1)
  return output


def create_model(x0: tf.Tensor, x1: tf.Tensor, time: tf.Tensor,
                 config: options.Options, net_flag='azadegan') -> tf.keras.Model:
  """Creates a frame interpolator model.

  The frame interpolator is used to warp the two images to the in-between frame
  at given time. Note that training data is often restricted such that
  supervision only exists at 'time'=0.5. If trained with such data, the model
  will overfit to predicting images that are halfway between the two inputs and
  will not be as accurate elsewhere.

  Args:
    x0: first input image as BxHxWxC tensor.
    x1: second input image as BxHxWxC tensor.
    time: ignored by film_net. We always infer a frame at t = 0.5.
    config: FilmNetOptions object.

  Returns:
    A tf.Model that takes 'x0', 'x1', and 'time' as input and returns a
          dictionary with the interpolated result in 'image'. For additional
          diagnostics or supervision, the following intermediate results are
          also stored in the dictionary:
          'x0_warped': an intermediate result obtained by warping from x0
          'x1_warped': an intermediate result obtained by warping from x1
          'forward_residual_flow_pyramid': pyramid with forward residual flows
          'backward_residual_flow_pyramid': pyramid with backward residual flows
          'forward_flow_pyramid': pyramid with forward flows
          'backward_flow_pyramid': pyramid with backward flows

  Raises:
    ValueError, if config.pyramid_levels < config.fusion_pyramid_levels.
  """
  if net_flag in ['both','filmNet']:
      if config.pyramid_levels < config.fusion_pyramid_levels:
        raise ValueError('config.pyramid_levels must be greater than or equal to '
                        'config.fusion_pyramid_levels.')

      x0_decoded = x0
      x1_decoded = x1

      image1 = tf.expand_dims(x0, 1)
      image2 = tf.expand_dims(x1, 1)
      input_img1 = tf.concat([image1, image2], axis=1)
      input_img2 = tf.concat([image1, image2], axis=1)
      image_pyramids_3D = [util.build_image_pyramid_3D(input_img1, config), util.build_image_pyramid_3D(input_img2, config)]
      # shuffle images
      image_pyramids = [
          util.build_image_pyramid(x0_decoded, config),
          util.build_image_pyramid(x1_decoded, config)
      ]

      # Siamese feature pyramids:
      extract = feature_extractor.FeatureExtractor('feat_net', config)
      extract_3D = feature_extractor.FeatureExtractor_3D('feat_net_3D', config)
      feature_pyramids = [extract(image_pyramids[0]), extract(image_pyramids[1])]
      feature_pyramids_3D = [extract_3D(image_pyramids_3D[0]), extract_3D(image_pyramids_3D[1])]
      predict_flow = pyramid_flow_estimator.PyramidFlowEstimator(
          'predict_flow', config)
      predict_flow_3D = pyramid_flow_estimator.PyramidFlowEstimator_3D('predict_flow_3D', config)

      # Predict forward flow.
      forward_residual_flow_pyramid = predict_flow(feature_pyramids[0],
                                                  feature_pyramids[1])
      # Predict backward flow.
      backward_residual_flow_pyramid = predict_flow(feature_pyramids[1],
                                                    feature_pyramids[0])
      forward_residual_flow_pyramid_3d = predict_flow_3D(feature_pyramids_3D[0], feature_pyramids_3D[1])
      backward_residual_flow_pyramid_3D = predict_flow_3D(feature_pyramids_3D[1], feature_pyramids_3D[0])

      # Concatenate features and images:

      # Note that we keep up to 'fusion_pyramid_levels' levels as only those
      # are used by the fusion module.
      fusion_pyramid_levels = config.fusion_pyramid_levels

      forward_flow_pyramid = util.flow_pyramid_synthesis(
          forward_residual_flow_pyramid)[:fusion_pyramid_levels]
      backward_flow_pyramid = util.flow_pyramid_synthesis(
          backward_residual_flow_pyramid)[:fusion_pyramid_levels]
      
      forward_flow_pyramid_3D = util.flow_pyramid_synthesis(forward_residual_flow_pyramid_3d)[:fusion_pyramid_levels]
      backward_flow_pyramid_3D = util.flow_pyramid_synthesis(backward_residual_flow_pyramid_3D)[:fusion_pyramid_levels]

      # We multiply the flows with t and 1-t to warp to the desired fractional time.
      #
      # Note: In film_net we fix time to be 0.5, and recursively invoke the interpo-
      # lator for multi-frame interpolation. Below, we create a constant tensor of
      # shape [B]. We use the `time` tensor to infer the batch size.
      mid_time = tf.keras.layers.Lambda(lambda x: tf.ones_like(x) * 0.5)(time)
      backward_flow = util.multiply_pyramid(backward_flow_pyramid, mid_time[:, 0])
      forward_flow = util.multiply_pyramid(forward_flow_pyramid, 1 - mid_time[:, 0])

      backward_flow_3D = util.multiply_pyramid(backward_flow_pyramid_3D, mid_time[:, 0])
      forward_flow_3D = util.multiply_pyramid(forward_flow_pyramid_3D, 1 - mid_time[:, 0])

      pyramids_to_warp = [
          util.concatenate_pyramids(image_pyramids[0][:fusion_pyramid_levels],
                                    feature_pyramids[0][:fusion_pyramid_levels]),
          util.concatenate_pyramids(image_pyramids[1][:fusion_pyramid_levels],
                                    feature_pyramids[1][:fusion_pyramid_levels])
      ]
      pyramids_to_warp_3D = [
          util.concatenate_pyramids(image_pyramids_3D[0][:fusion_pyramid_levels], feature_pyramids_3D[0][:fusion_pyramid_levels]),
          util.concatenate_pyramids(image_pyramids_3D[1][:fusion_pyramid_levels], feature_pyramids_3D[1][:fusion_pyramid_levels])]
      # Warp features and images using the flow. Note that we use backward warping
      # and backward flow is used to read from image 0 and forward flow from
      # image 1.
      forward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[0], backward_flow)
      backward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[1], forward_flow)

      forward_warped_pyramid_3D = util.pyramid_warp(pyramids_to_warp_3D[0], backward_flow)
      backward_warped_pyramid_3D = util.pyramid_warp(pyramids_to_warp_3D[1], forward_flow)

      aligned_pyramid = util.concatenate_pyramids(forward_warped_pyramid,
                                                  backward_warped_pyramid)
      aligned_pyramid = util.concatenate_pyramids(aligned_pyramid, backward_flow)
      aligned_pyramid = util.concatenate_pyramids(aligned_pyramid, forward_flow)

      aligned_pyramid_3D = util.concatenate_pyramids(forward_warped_pyramid_3D,
                                                  backward_warped_pyramid_3D)
      aligned_pyramid_3D = util.concatenate_pyramids(aligned_pyramid_3D, backward_flow_3D)
      aligned_pyramid_3D = util.concatenate_pyramids(aligned_pyramid_3D, forward_flow_3D)

      fuse = fusion.Fusion('fusion', config)
      prediction = fuse(aligned_pyramid)
      output_color = prediction[..., :3]

      fuse_3D = fusion.Fusion('fusion', config)
      prediction_3D = fuse_3D(aligned_pyramid_3D)

      output_color_3D = prediction_3D[..., :3]
  else:
      output_color = x0

  ##################################
  if net_flag=='azadegan':
     output2 = create_azadegan_model(x0, x1, options)
     outputs = {'image': output2}
  elif net_flag=='both':
    output2 = create_azadegan_model(x0, x1, options)
    final_output = tf.add(output2,output_color)
    output_color = final_output
    outputs = {'image': output_color, 'image_azadNet':output2,'final_output':final_output}
  else:
    outputs = {'image': output_color}

  if config.use_aux_outputs and net_flag in ['both','filmNet']:
    outputs.update({
        'x0_warped': forward_warped_pyramid[0][..., 0:3],
        'x1_warped': backward_warped_pyramid[0][..., 0:3],
        'forward_residual_flow_pyramid': forward_residual_flow_pyramid,
        'backward_residual_flow_pyramid': backward_residual_flow_pyramid,
        'forward_flow_pyramid': forward_flow_pyramid,
        'backward_flow_pyramid': backward_flow_pyramid,
    })

  model = tf.keras.Model(
      inputs={
          'x0': x0,
          'x1': x1,
          'time': time
      }, outputs=outputs)
  return model
