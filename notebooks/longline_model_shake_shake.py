# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import datetime
import dateutil.parser
from glob import glob
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import numpy as np
import os
import time
import pandas as pd
import pandas.io.gbq
import pickle
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from keras import backend

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %matplotlib inline

# This is a DAGEROUS workaround to having two KMP libraries linked
# Would be better to figure out conflict
import sys
sys.path.append('..')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

plt.rcParams['figure.figsize'] = [10, 5]

from track_based_models import longline_sets_models
from track_based_models import util as tbutil
from track_based_models.util import minute, hour

import pyseas
from pyseas import maps, cm, styles, util
from pyseas.contrib import plot_tracks
from pyseas.maps import scalebar, core, rasters, ticks
import imp

def reload():
    imp.reload(util)
    imp.reload(ticks)
    imp.reload(scalebar)
    imp.reload(cm)
    imp.reload(styles)
    imp.reload(rasters)
    imp.reload(core)
    imp.reload(maps)
    imp.reload(plot_tracks)
    imp.reload(pyseas)
reload()

# +
# Step 1. Figure out where we are going to get the train and test data
# Break up by boat – if we validate using the same boat validated by
# two different people it will wildly optimistic

# This assumes the birdlife repo is installed alongside this!

core_paths = sorted((glob("../../birdlife/labeled_data/without_logbook_data/*.json") +
                glob("../../birdlife/labeled_data/with_logbook_data/*.json")))

all_paths = core_paths + \
            glob("../../birdlife/unlabelled_data/*.json") + \
            glob("../../birdlife/allboats/*.json")
# -

paths_to_ssvid = {x : os.path.basename(x).split('_')[0] for x in all_paths}
all_ssvid = ssvid = sorted(set(paths_to_ssvid[x] for x in core_paths))
','.join(['"{}"'.format(x) for x in all_ssvid])

query = '''
select * from 
(select id as track_id, split(id, "-")[ORDINAL(1)] AS mmsi, * from 
 `machine_learning_dev_ttl_120d.single_track_features_v20200625_*`
) features
where 
mmsi in ({}) -- and EXTRACT(minute FROM timestamp) = 0
order by timestamp
'''.format(
','.join(['"{}"'.format(x) for x in all_ssvid])
)
all_features = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

print(len(set(all_features.mmsi))) # 13
all_features['timestamp'] = all_features['timestamp'].dt.tz_convert(None)

# +
for mmsi in sorted(set(all_features.mmsi)):
    df = all_features[all_features.mmsi == mmsi]
    tid = sorted(set(df.track_id))
    print(mmsi, len(tid))
    if len(tid) != 1:
        for t in tid:
            df2 = df[df.track_id == t]
            print(t, len(df2), df2.timestamp.min().date(), df2.timestamp.max().date())
# Only problematic track_id appears to be 659283000-2017-07-19T19:23:14.000000Z-2017-07-19


all_features = all_features[all_features.track_id != '659283000-2017-07-19T19:23:14.000000Z-2017-07-19']
all_features['ssvid'] = all_features.mmsi
# -

all_feature_ssvid = sorted(set(all_features.ssvid))
for x in sorted(core_paths):
    if paths_to_ssvid[x] not in all_feature_ssvid:
        print(x, 'missing')

all_features['speed'] = all_features['speed_knots']

# ## Cross Validation

# +
# import imp, track_based_models.longline_sets_models
# imp.reload(track_based_models.util)
# imp.reload(track_based_models.base_model)
# imp.reload(track_based_models.single_track_model)
# imp.reload(track_based_models.longline_sets_models)
# from track_based_models.longline_sets_models import LonglineSetsModelV4 as Model

# # objs = []
# # for p in core_paths:
# #     try: 
# #         objs.append(Model.load_data(p, all_features))
# #     except KeyError as err:
# #         print(p, "failed", err)


# +
import imp, track_based_models.longline_sets_models
imp.reload(track_based_models.shake_shake)
imp.reload(track_based_models.util)
imp.reload(track_based_models.base_model)
imp.reload(track_based_models.single_track_model)
imp.reload(track_based_models.longline_sets_models)
from track_based_models.longline_sets_models import LonglineSetsModel12 as Model


max_epochs = 40
keep_fracs = [1, 0.75, 0.5, 0.25]
batch_size = 64

f1_scores = []
class_b_f1_scores = []
david_class_b_f1_scores = []

def test_model(mdl, times, features, labels, defined, verbose=False):
    predictions = mdl.predict(features) > 0.5
#     print(np.shape(times), np.shape(features), np.shape(labels))
#     print(labels.shape, predictions.shape, defined.shape)
    precision, recall, fscore, support = precision_recall_fscore_support(labels.ravel(), predictions.ravel(), 
                                                                         sample_weight=defined.ravel())
    accuracy = accuracy_score(labels.ravel(), predictions.ravel(), sample_weight=defined.ravel())
    auc = roc_auc_score(labels.ravel(), predictions.ravel(), sample_weight=defined.ravel())
    if verbose:
        print("model accuracy:", accuracy)
        print("model precision:", precision[1])
        print("model recall:", recall[1])
        print("model f1-score:", fscore[1])
        print("model set support:", support[1])
        print()
    return accuracy, precision[1], recall[1], fscore[1], auc, support[1]
    
# Test the model
# print('testing')
# test_model(mdl, vtimes, vfeatures, vlabels, vdefined)



train_paths = sorted([x for x in core_paths if paths_to_ssvid[x]])
objs = [Model.load_data(p, all_features) for p in train_paths]


f1_scores = []
results_list = []
np.random.seed(888)
folder = GroupKFold(n_splits=5)
for i, (train_index, test_index) in list(enumerate(
    folder.split(train_paths, train_paths, [paths_to_ssvid[x] for x in train_paths]))):
    print()
    print("Fold", i)
    print(sorted(set([paths_to_ssvid[train_paths[j]] for j in test_index])))
    train_objs = [objs[j] for j in train_index]
    test_objs = [objs[j] for j in test_index]
    
    mdl = Model(width=Model.internal_time_points)
    
#   Train the model    
    print('generating training data')
    xtimes, xfeatures, xlabels, _, xdefined = mdl.generate_inputs(train_objs, 
                                                                  samples_per_obj=400, 
                                                                  keep_fracs=keep_fracs,
                                                                  offsets=[0],
                                                                  extra_time_deltas=0,
                                                                  noise=0)
    print('generating test data')
    vtimes, vfeatures, vlabels, _, vdefined = mdl.generate_inputs(test_objs, 
                                                                samples_per_obj=100, 
                                                                noise=0,
                                                                extra_time_deltas=0)
    
    print(np.shape(xtimes), np.shape(xfeatures), np.shape(xlabels), 
          np.shape(xdefined.shape))

    results = []
    xweights = xdefined# * ((xlabels == 0) + 2 * (xlabels == 1))
    for i in range(max_epochs):
        mdl.fit(xfeatures, xlabels, initial_epoch=i, epochs=i+1, 
                              sample_weight=xweights, batch_size=batch_size)
        accuracy, precision, recall, fscore, auc, support = test_model(mdl, vtimes, vfeatures, vlabels, vdefined)
        print('Epoch:', i, fscore, precision, recall, accuracy, auc, support)
        results.append((accuracy, precision, recall, fscore, auc, support))
        backend.set_value(mdl.optimizer.lr, 
            0.9 * backend.get_value(mdl.optimizer.lr))
    
    # Test the model
    print('testing')
    fscore = test_model(mdl, vtimes, vfeatures, vlabels, vdefined)
    f1_scores.append(fscore[3])
    results_list.append(results)

        
print("Score derived from random sampling:")
print("  Average F1-Score:", np.mean(f1_scores))
# 0.9167440972301417 11, one sided
# 0.910695424990904  5, one sided
#   Average F1-Score: 0.9223069457487714 # 30 epochs 5, one-sided 0.9178247286426837 0.9144560341081686 
# 0.9353990721786382   Average F1-Score: 0.9360450419189241
# -
import keras.layers
keras.layers.ReLU

from collections import Counter
Counter(objs[-1].fishing).most_common()

results_2 = np.array(results_list)


# +

def plot(data, ndx=3, lbl='f1', label='', show_individual=True, color='k'):
    if show_individual:
        for x in data:
            plt.plot(x[:, ndx], '-', linewidth=0.25, color=color)
    plt.plot(data.mean(axis=0)[:, ndx], label=label, color=color)
    
# plot(results_1, label='f1', color='c')
plot(results_2, label='f1', color='b')
# plot(results_3, label='f1', color='r')
# plot(results_4, label='f1', color='orange')
# plot(results_5, label='f1', color='purple')
# plot(results_6, label='f1', color='g')




plt.grid()
plt.legend()
plt.ylim(0.6, 0.9)
# -

# Train the model on all of the data
print("generating data")
mdl_all = Model()
times, features, labels, _, defined = mdl_all.generate_inputs(objs, samples_per_obj=400, keep_fracs=keep_fracs,
                                                          offsets=[0],
                                                         extra_time_deltas=0,
                                                         noise=0)
print("n_windows of data", len(features))
print("training")
epochs = 30
for i in range(epochs):
    mdl_all.fit(features, labels, initial_epoch=i, epochs=i+1, 
                          sample_weight=defined, batch_size=batch_size)
    backend.set_value(mdl_all.optimizer.lr, 
        0.9 * backend.get_value(mdl_all.optimizer.lr))

# 1. (X) Train on all.
# 2. (X) Save as h5 file
# 3. (X) Get standalone working.
# 4. ( ) Run features for all of 2017
# 5. ( ) Copy run iference using the Keras version -- should be easier!
# 6. ( ) Run on all longliners for 2017.
# 7. ( ) Have someone spot check.
# 8. ( ) Copy over to loitering.

# +
# mdl_all.save('longline_set_model_12_v20200702.h5')

# +
# # !gsutil cp longline_set_model_12_v20200702.h5 gs://machine-learning-dev-ttl-120d/

# +
import imp, track_based_models.longline_sets_models
imp.reload(track_based_models.util)
imp.reload(track_based_models.base_model)
imp.reload(track_based_models.single_track_model)
imp.reload(track_based_models.longline_sets_models)
from track_based_models.longline_sets_models import LonglineSetsModelV4 as Model

# reload()
# mdl2 = Model.load('longline_set_model_12_v20200702.h5')

# +
# Buidl dictionary to see this.
# then plot
# Suspect we have off by N error where N is ~12, possibly related to using hour rather 
# rather than 5 minute intervals
# -

sorted(set([obj.ssvid.iloc[0] for obj in objs if len(obj.ssvid)]))

i = 0
test_ssvid = ['416002659', '601061600']
ssvid = test_ssvid[i]
for obj in objs:
    if not len(obj.ssvid):
        continue
    if obj.ssvid.iloc[0] == ssvid:
        break
test_data = mdl.util.features_to_data(all_features[all_features.ssvid == ssvid])
results = mdl.predict_from_data(test_data)
data_map = {}
for x in test_data.itertuples():
    data_map[x.timestamp] = x

is_fishing = (obj.fishing == 1)
is_fishing[obj.fishing == 2] = np.nan
plt.plot(obj.timestamp, is_fishing)
plt.plot(results[0], results[1], 'y.')
plt.plot(results[0], results[1] > 0.5, 'r.')
plt.xlim(obj.timestamp.iloc[0], obj.timestamp.iloc[-1])

# +
mask = (results[0] >= obj.timestamp.iloc[0]) & (results[0] <= obj.timestamp.iloc[-1])
lons = []
lats = []
vals = []
lbls = []
times = []
speeds = []
mask = np.arange(len(results[0])) > len(results[0])- 500
for t, v in zip(results[0][mask], results[1][mask]):
    if t in data_map:
        d = data_map[t]
        times.append(t)
        lons.append(d.lon)
        lats.append(d.lat)
        speeds.append(d.speed)
        vals.append(v)
lons = np.array(lons)
lats = np.array(lats)
vals = np.array(vals)

plt.figure(figsize=(12, 16))
with pyseas.context(styles.light):
    info = plot_tracks.plot_fishing_panel(times, lons, lats, vals > 0.5,
                             plots = [
            {'label' : 'speed (knots)', 'values' : speeds, 
                'min_y' : 0},                       
                             ],
                             map_ratio=6,
                             annotations=0)
    
plt.show()

# with pyseas.context(styles.light):
#     info = plot_tracks.plot_fishing_panel(obj.timestamp, obj.lon, obj.lat, obj.fishing == 0,
#                              plots = [
# #             {'label' : 'speed (knots)', 'values' : speeds, 
# #                 'min_y' : 0},                       
#                              ],
#                              map_ratio=6,
#                              annotations=0)
# -

# ## Compare With Original

with open('../data/birdlife_ais_data.pkl', 'rb') as f:
    birdlife_ais_data = pickle.load(f)


# +
def compare(obj, times, predictions):
#     p = obj['path'].replace('../..', '..')
    p = obj['path']
    with open(p) as f:
        orig_obj = json.load(f)
        orig = pd.DataFrame({
            'timestamp' : orig_obj['timestamps'],
            'fishing' : orig_obj['fishing'],
            'mmsi' : [orig_obj['mmsi']] * len(orig_obj['timestamps']),
            'lat' :  orig_obj['lats'],
            'lon' : orig_obj['lons'],
        })
    orig['timestamp'] = [dateutil.parser.parse(x) for x in orig['timestamp']]
    data = add_predictions(orig, times, predictions)
    assert len(data) == len(orig.timestamp)#, (len(data), len(orig['timestamp']))
    inferred = data.inferred_setting.values
    annotated = data.fishing.values
    defined = (data.fishing != 0)
    mask = ~np.isnan(inferred) & ~np.isnan(annotated) & defined
    match = (inferred == 1) == (annotated == 1)
    true_pos = (match & (annotated == 1))[mask].sum()
    all_true = (inferred == 1)[mask].sum()
    all_pos  = (annotated == 1)[mask].sum()
    prec = true_pos / all_pos
    rec = true_pos / all_true
    f1 = 2 / (1 / prec + 1 / rec)
    acc = match[mask].sum() / mask.sum()
    orig_obj['timestamps'] = [dateutil.parser.parse(x) for x in orig_obj['timestamps']]
    return orig_obj, data, {'accuracy' : acc, 'F1' : f1, 'precision' : prec, 'recall' : rec}

def add_predictions(data, times, predictions):
    preds = np.empty(len(data), dtype=float)
    assert len(times) == len(predictions)
    timestamps = [x.to_pydatetime() for x in data.timestamp]
    assert util.is_sorted(timestamps)
    preds.fill(np.nan)
    for t, p in zip(times, predictions):
        t0 = t - datetime.timedelta(seconds=Model.delta // 2)
        t1 = t + datetime.timedelta(seconds=Model.delta // 2)
        i0 = np.searchsorted(timestamps, t0, side='left')
        i1 = np.searchsorted(timestamps, t1, side='right')
        preds[i0:i1] = p
    data = data.copy()
    data['inferred_setting'] = preds
    return data


# +
# all_features['timestamp'] = all_features.timestamp.dt.tz_convert(None)
# -

len(all_features)

# +
import imp, track_based_models.longline_sets_models
imp.reload(track_based_models.util)
imp.reload(track_based_models.base_model)
imp.reload(track_based_models.single_track_model)
imp.reload(track_based_models.longline_sets_models)
from track_based_models.longline_sets_models import LonglineSetsModelV2 as Model

metrics_list = []
objs_list = []
for base_obj in birdlife_ais_data:
    ssvid = base_obj['mmsi']
    t0 = base_obj['timestamps'][0]
    t1 = base_obj['timestamps'][-1]
    mdl = LonglineSetsModelV2.load('gs://machine-learning-dev-ttl-120d/longline_model_v20190615.h5')
    data = mdl.util.features_to_data(all_features, ssvid, t0, t1)
    times, predictions = mdl2.predict_set_times(data)
    orig_obj, orig, metrics = compare(base_obj, times, predictions)
    metrics_list.append(metrics)
    objs_list.append((orig_obj, orig, times, predictions))
    print('.', end='')
# -

print(len(metrics_list), np.mean([x['F1'] for x in metrics_list]))
# 0.8587006713736698 0.8765266861195162 0.8528620543033096 0.8507986578373596 0.8545812824063037 0.8520529794641322

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.figure(figsize=(12, 6))
orig_obj, orig, times, predictions = objs_list[8]
mask = (orig['fishing'] != 0)
plt.plot(orig['timestamp'][mask], orig['fishing'][mask] == 1, 'r.')
plt.plot(times, predictions)

# ## Examine Dataflow Inferred Data

ssvid_list = []
for base_obj in birdlife_ais_data:
    ssvid_list.append(base_obj['mmsi'])

query = '''
select a.*, lon, lat, speed_knots, split(id, "-")[ORDINAL(1)] AS ssvid
from `machine_learning_dev_ttl_120d.sets_detections_v20200702_*` a
join `machine_learning_dev_ttl_120d.single_track_features_v20200625_*`
using(id, timestamp)
where split(id, "-")[ORDINAL(1)] in ({})
order by timestamp
'''.format(','.join(['"{}"'.format(x) for x in ssvid_list]))
print(query)
pipeline_inferred = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

pipeline_inferred['timestamp'] = pipeline_inferred.timestamp.dt.tz_convert(None)

ssvids = sorted(set(pipeline_inferred.id))

# +
i = 1
obj = objs[i]
ssvid = paths_to_ssvid[all_paths[i]]

df = pipeline_inferred[(pipeline_inferred.ssvid == ssvid)]
mask = (df.timestamp > obj.timestamp.iloc[0]) & (df.timestamp < obj.timestamp.iloc[-1])
df = df[mask]

is_fishing = (obj.fishing == 1)
# is_fishing[obj.fishing == 2] = np.nan
plt.plot(obj.timestamp, is_fishing)
plt.plot(df.timestamp, df.value > 0.5, 'r.')
plt.xlim(obj.timestamp.iloc[0], obj.timestamp.iloc[-1])
# -

len(ssvid)

# +
from pyseas import maps, cm, styles, util
from pyseas.contrib import plot_tracks
from pyseas.maps import scalebar, core, rasters, ticks
import imp

def reload():
    imp.reload(util)
    imp.reload(ticks)
    imp.reload(scalebar)
    imp.reload(cm)
    imp.reload(styles)
    imp.reload(rasters)
    imp.reload(core)
    imp.reload(maps)
    imp.reload(plot_tracks)
    imp.reload(pyseas)
reload()

import pyseas
from pyseas import maps, styles
from pyseas.contrib import plot_tracks
from scipy.signal import medfilt

ssvid = sorted(set(pipeline_inferred.id))
for x in ssvid:
    df = pipeline_inferred[(pipeline_inferred.id == x) &
                           (pipeline_inferred.timestamp >= datetime.datetime(2018, 4, 1)) &
                           (pipeline_inferred.timestamp < datetime.datetime(2018, 5, 1)) ]
    if len(df) < 10:
        continue
    plt.figure(figsize=(12, 8))
    with pyseas.context(styles.light):
        plot_tracks.plot_fishing_panel(df.timestamp, df.lon,
                                 df.lat, np.roll(df.value.values,12 ) > 0.5,
                                 plots = [
                {'label' : 'speed (knots)', 'values' : medfilt(df.speed_knots.values,11), 
                    'min_y' : 0},                       
                                 ],
                                 map_ratio=6,
                                 annotations=5)
        plt.title(x)
        
    
        plt.show()
# info.map_ax.set_extent((50, 60, 0, 10), crs=maps.identity)
# -

# ## Inspect 20 random mmsi

query = '''
with ssvid_list as (
select distinct cast(vessel_mmsi as string) as ssvid from `birdlife.birdlife_set_counts*`
order by farm_fingerprint(ssvid)
limit 20
)

select a.*, lon, lat, speed_knots, split(id, "-")[ORDINAL(1)] AS ssvid from 
`machine_learning_dev_ttl_120d.sets_detections_v20200702_*` a
join `machine_learning_dev_ttl_120d.single_track_features_v20200625_201704*`
using(id, timestamp)
where split(id, "-")[ORDINAL(1)] in (select * from ssvid_list)
order by timestamp
'''.format(','.join(['"{}"'.format(x) for x in ssvid_list]))
print(query)
pipeline_inferred = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

# +
ssvid = sorted(set(pipeline_inferred.ssvid))

for x in ssvid:
    df = pipeline_inferred[(pipeline_inferred.ssvid == x)]

    plt.figure(figsize=(12, 8))
    with pyseas.context(styles.light):
        info = plot_tracks.plot_fishing_panel(df.timestamp, df.lon,
                                 df.lat, df.value.values > 0.5,
                                 plots = [
                {'label' : 'speed (knots)', 'values' : df.speed_knots.values, 
                    'min_y' : 0},                       
                                 ],
                                 map_ratio=6,
                                 annotations=5)
        plt.title(x)
        # From https://nbviewer.jupyter.org/gist/pelson/626b15ffc411a359381e
        cmap = plt.get_cmap('Blues')
        norm = mpcolors.Normalize(0, 10000)

        for letter, level in [
                              ('L', 0),
                              ('K', 200),
                              ('J', 1000),
                              ('I', 2000),
                              ('H', 3000),
                              ('G', 4000),
                              ('F', 5000),
                              ('E', 6000),
                              ('D', 7000),
                              ('C', 8000),
                              ('B', 9000),
                              ('A', 10000)]:
            bathym = NaturalEarthFeature(name='bathymetry_{}_{}'.format(letter, level),
                                         scale='10m', category='physical')
            info.map_ax.add_feature(bathym, facecolor=cmap(norm(level)), edgecolor='face', zorder=0)
            
            
        a, b, c, d = info.map_ax.get_extent(crs=maps.identity)
        print(a, b, c, d)
        info.map_ax.set_extent((a - 5, b + 5, c - 5, d + 5), crs=maps.identity)
        maps.add_gridlines()
        maps.add_gridlabels()
        
        plt.show()
        print('\n\n\n\n')
# -

pipeline_inferred.groupby('ssvid').count().timestamp.max()

query = """
with 
--
-- Limit to only pelagic longlines. That is,
-- longlines that fish in water deeper than 200 meters 
-- on average
longlines as (
select ssvid from `gfw_research.vi_ssvid_v20200701`
where activity.overlap_hours_multinames < 24
and best.best_vessel_class = 'drifting_longlines'
and activity.avg_depth_fishing_m < -200
),

lag_lead as (
    select 
        id as ssvid,
        timestamp,
        ifnull(value,0) value,
        lead(timestamp,1) over (partition by id order by timestamp) next_timestamp,
        lag(timestamp,1) over (partition by id order by timestamp) last_timestamp,
        lead(value,1) over (partition by id order by timestamp) next_value,
        lag(value,1) over (partition by id order by timestamp) last_value
    from `machine_learning_dev_ttl_120d.sets_detections_v20200702_*`
    where split(id, "-")[ORDINAL(1)]  in (select ssvid from longlines)
),
--
-- eliminate single points between others
-- 
--
-- If a single negative is between two positives, eliminate. 
eliminated_singlepoints_pos as ( 
select * except(value, next_value, last_value),
case when next_value >= .5 and last_value >= .5  and value < .5 then 1
else value end value
from lag_lead),
--
-- Now, lag lead again to get next and last *after* eliminated positives
-- between two negatives
--
lag_led_weliminated_pos as (
select *, 
lead(value,1) over (partition by ssvid order by timestamp) next_value,
lag(value,1) over (partition by ssvid order by timestamp) last_value
from eliminated_singlepoints_pos),
--
-- Now eliminate points that are single positives points between two negative points
--
eliminated_singlepoints as ( 
select * except(value, next_value, last_value),
case when next_value < .5 and last_value < .5  and value >= .5 then 0
else value end value
from lag_led_weliminated_pos),
--
-- Lag lead again because value has changed!
--
lag_led_weliminated as (
select *, 
lead(value,1) over (partition by ssvid order by timestamp) next_value,
lag(value,1) over (partition by ssvid order by timestamp) last_value
from eliminated_singlepoints),
-- get starts and ends of sets
--
set_start_or_end as 
(select *,
last_value < .5 and value >=.5 set_start,
next_value < .5 and value >=.5 set_end
from lag_led_weliminated ),
--  
-- 
-- 
set_stats as 
(select ssvid,
timestamp_sub(timestamp, INTERVAL 30 MINUTE)  as set_start_timestamp,
timestamp_add(set_end_timestamp, INTERVAL 30 MINUTE)  as set_end_timestamp,
timestamp_diff(set_end_timestamp, timestamp, second)/3600 + 1 set_hours,
set_end_next
from 
  (
  select 
    *,
    lead(timestamp,1) over (partition by ssvid order by timestamp) set_end_timestamp,
    lead(set_end,1) over (partition by ssvid order by timestamp) set_end_next
  from 
    set_start_or_end
  where set_start or set_end
  )
where set_start
and next_timestamp is not null
and set_end_timestamp is not null
order by ssvid, timestamp)
select * except(set_end_next) from set_stats
where set_end_next
"""
set_stats = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

vals, bins, patches = plt.hist(set_stats.set_hours, bins=24, range=(0, 24))

plt.scatter(np.arange(0.5, 24.5), vals)
plt.xlim(0, 24)
plt.ylabel('number of set')
plt.xlabel('hours')
plt.title('sets_detections_v20200702_2017*')



bins

# ## Commands to Run in the Pipeline
#
#
# This is done on the `bathymetry` branch.
#
#     python -m pipe_features.create_loitering_features \
#         --start_date 2015-01-01 \
#         --end_date 2020-02-29 \
#         --source_table pipe_production_v20190502.position_messages_ \
#         --sink_table machine_learning_dev_ttl_120d.test_longline_features_mmsi_v20190502_ \
#         --project world-fishing-827 \
#         --temp_location gs://machine-learning-dev-ttl-30d/scratch/features \
#         --job_name smoke-test-loitering-features \
#         --setup_file ./setup.py \
#         --requirements_file ./requirements.txt \
#         --runner DataflowRunner \
#         --max_num_workers 200 \
#         --worker_machine_type=custom-1-13312-ext \
#         --id_filter_query "SELECT distinct ssvid FROM \`world-fishing-827.gfw_research.vi_ssvid_byyear_v20190430\` WHERE best.best_vessel_class = 'drifting_longlines'"
#
#
#     python -m pipe_features.create_loitering_features \
#         --start_date 2015-01-01 \
#         --end_date 2020-02-29 \
#         --source_table pipe_production_v20190502.position_messages_ \
#         --sink_table machine_learning_dev_ttl_120d.test_longline_training_features_mmsi_v20190502_ \
#         --project world-fishing-827 \
#         --temp_location gs://machine-learning-dev-ttl-30d/scratch/features \
#         --job_name smoke-test-loitering-features \
#         --setup_file ./setup.py \
#         --requirements_file ./requirements.txt \
#         --runner DataflowRunner \
#         --max_num_workers 200 \
#         --worker_machine_type=custom-1-13312-ext \
#         --id_filter_query '"416002616","416002616","416002616","416002616","416002659","416002659","416002659","416004865","416005500","416005500","416005500","416085700","416085700","416768000","416768000","416826000","416826000","416826000","416874000","416874000","431704220","431704220","432298000","432298000","432298000","601061600","601061600","601061600","601061600","601274700","601274700","601274700","601274700","659283000","659283000","659283000","659283000"'
#
#
#     python -m pipe_features.sets_inference \
#             --feature_table machine_learning_dev_ttl_120d.test_longline_training_features_mmsi_v20190502_ \
#             --model_path 'gs://machine-learning-dev-ttl-120d/longline_model_v20200403.h5' \
#             --results_table machine_learning_dev_ttl_120d.longline_model_v2_20200403_\
#             --start_date 2017-01-01 \
#             --end_date 2018-12-31 \
#             --temp_location=gs://machine-learning-dev-ttl-30d/scratch/nnet-char \
#             --runner DataflowRunner \
#             --project=world-fishing-827 \
#             --job_name=sets-test-2 \
#             --max_num_workers 100 \
#             --requirements_file=./requirements.txt \
#             --setup_file=./setup.py \
#             --worker_machine_type=custom-1-13312-ext \
#             --target_inference_width 1
#
#
#
#     python -m pipe_features.sets_inference \
#             --feature_table machine_learning_dev_ttl_120d.test_longline_features_mmsi_v20190502_ \
#             --model_path 'gs://machine-learning-dev-ttl-120d/longline_model_v20200403.h5' \
#             --results_table machine_learning_dev_ttl_120d.longline_model_v2_20200403_all_\
#             --start_date 2017-01-01 \
#             --end_date 2018-12-31 \
#             --temp_location=gs://machine-learning-dev-ttl-30d/scratch/nnet-char \
#             --runner DataflowRunner \
#             --project=world-fishing-827 \
#             --job_name=sets-test-2 \
#             --max_num_workers 100 \
#             --requirements_file=./requirements.txt \
#             --setup_file=./setup.py \
#             --worker_machine_type=custom-1-13312-ext \
#             --target_inference_width 1

# ## Old Stuff Below This Point

pipeline_inferred['timestamp'] = pipeline_inferred.timestamp.dt.tz_convert(None)

# +

metrics_list = []
objs_list = []
for base_obj in birdlife_ais_data:
    ssvid = base_obj['mmsi']
    t0 = base_obj['timestamps'][0]
    t1 = base_obj['timestamps'][-1]
    mask = ((pipeline_inferred.id == ssvid) & 
            (t0 <= pipeline_inferred.timestamp) & 
            (pipeline_inferred.timestamp <= t1))
    times = pipeline_inferred.timestamp[mask]
    predictions = pipeline_inferred.value[mask]
    orig_obj, orig, metrics = compare(base_obj, times, predictions)
    metrics_list.append(metrics)
    objs_list.append((orig_obj, orig, times, predictions))
    print('.', end='')
# -

print(len(metrics_list), np.mean([x['F1'] for x in metrics_list])) # 0.8509712217034556

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.figure(figsize=(12, 6))
orig_obj, orig, times, predictions = objs_list[8]
mask = (orig['fishing'] != 0)
plt.plot(orig['timestamp'][mask], orig['fishing'][mask] == 1, 'r.')
plt.plot(times, predictions)

orig.columns

for orig_obj, orig, times, predictions in objs_list:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.plot(orig.lon, orig.lat, 'k', linewidth=0.2)
    mask = orig.inferred_setting != 0
    lon = orig.lon.copy()
    lat = orig.lat.copy()
    lon[mask] = np.nan
    lat[mask] = np.nan
    ax1.plot(lon, lat, 'c')
    mask = orig.inferred_setting != 1
    lon = orig.lon.copy()
    lat = orig.lat.copy()
    lon[mask] = np.nan
    lat[mask] = np.nan
    ax1.plot(lon, lat, 'r')
    ax1.set_xlabel('longitude')
    ax1.set_ylabel('latitiude')
    
    mask = (orig['fishing'] != 0)
    ax2.plot(orig['timestamp'][mask], orig['fishing'][mask] == 1, 'kx', label='annotated')
    ax2.plot(times, predictions, 'c-', label='inferred',  markersize=6)
    ax2.legend()
    ax2.set_xlabel('time')
    ax2.set_ylabel('is fishing')
    
    ax1.set_title(str(orig.mmsi[0]) + ' tracks')
    ax2.set_title(str(orig.mmsi[0]) + ' time comparison')
    plt.tight_layout()

# # Hawaii fleet

query = '''
select * from `machine_learning_dev_ttl_120d.fishing_detection_V20190614_hawaii_b_*`
order by timestamp
'''
pipeline_hawaii_inferred = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

query = '''
SELECT * FROM `world-fishing-827.birdlife.hawaii_longline_predicted_sets`
order by set_start'''
old_hawaii_inferred = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

ha_ssvid = sorted(set(pipeline_hawaii_inferred.id))
old_hawaii_inferred.dtypes

for ssvid in ha_ssvid:
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    mask = pipeline_hawaii_inferred.id == ssvid
    ax1.plot(pipeline_hawaii_inferred[mask].timestamp, pipeline_hawaii_inferred[mask].value, 'k', label='pipeline')
    mask = old_hawaii_inferred.vessel_mmsi.astype(str) == ssvid
    for x in old_hawaii_inferred[mask].itertuples():
        ax1.plot([x.set_start, x.set_end], [1, 1], 'r')
    ax1.set_xlabel('time')
    ax1.set_ylabel('is fishing')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(datetime.datetime(2017, 4, 7), datetime.datetime(2017, 4, 21))


