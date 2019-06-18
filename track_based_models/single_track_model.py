from .base_model import BaseModel

class SingleTrackModel(BaseModel):

    data_source_lbl = None 
    data_target_lbl = None
    data_defined_vals = None
    data_true_vals = None

    def create_features_and_times(self, data, angle=77):
        t, xi, y, label_i, defined_i = self.build_features(data, skip_label=True)
        min_ndx = 0
        max_ndx = len(t) - self.time_points
        features = []
        for i in range(min_ndx, max_ndx):
            raw_features = y[i:i+self.time_points]
            features.append(self.cook_features(raw_features, angle=angle, noise=0)[0])
        times = t[self.time_points//2:-self.time_points//2]
        return features, times

    @classmethod
    def build_features(cls, obj, delta=None, skip_label=False, keep_frac=1.0):
        n_pts = len(obj['lat'])
        assert 0 < keep_frac <= 1, 'keep frac must be between 0 and 1'
        if keep_frac == 1:
            mask = None
        else:           
            # Build a random mask with probability keep_frac. Force
            # first and last point to be true so the time frame
            # stays the same.
            mask = np.random.uniform(0, 1, size=[n_pts]) < keep_frac
            mask[0] = mask[-1] = True 
            
        assert np.isnan(obj['speed']).sum() == np.isnan(obj['course']).sum() == 0, (
                'null values are not allow in the data, please filter first')

        v = np.array(obj['speed'])
        # Replace missing speeds with arbitrary 3.5 (between setting and hauling)
        # TODO: use implied speed instead
        v[np.isnan(v)] = 3.5
        obj['speed'] = v
        xi, speeds = lin_interp(obj, 'speed', delta=delta, mask=mask)
        y0 = speeds
        #
        _, cos_yi = lin_interp(obj, 'course', delta=delta, mask=mask, func=cos_deg)
        _, sin_yi = lin_interp(obj, 'course', delta=delta, mask=mask, func=sin_deg)
        angle_i = np.arctan2(sin_yi, cos_yi)
        y1 = angle_i
        #
        _, y2 = lin_interp(obj, 'lat', delta=delta, mask=mask)
        # Longitude can cross the dateline, so interpolate useing cos / sin
        _, cos_yi = lin_interp(obj, 'lon', delta=delta, mask=mask, func=cos_deg)
        _, sin_yi = lin_interp(obj, 'lon', delta=delta, mask=mask, func=sin_deg)
        y3 = np.degrees(np.arctan2(sin_yi, cos_yi))
        # delta times
        xp = util.compute_xp(obj, mask)
        dts = util.delta_times(xi, xp)
        y4 = dts
        if 'min_dt_min' in obj:
            dts += lin_interp(obj, 'min_dt_min', mask=None) * 60
        # Times
        t0 = obj['timestamp'].iloc[0]
        t = [(t0 + datetime.timedelta(seconds=delta * i)) for i in range(len(y1))]
        y = np.transpose([y0, y1, y2, y3, y4])
        #
        # Quick and dirty nearest neighbor (only works for binary labels I think)
        if skip_label:
            label_i = defined_i = None
        else:
            obj['is_defined'] = [w in cls.data_defined_vals for w in obj[cls.data_source_lbl]]
            obj[cls.data_target_lbl] = [w in cls.data_true_vals for w in obj[cls.data_source_lbl]]
            _, raw_label_i = lin_interp(obj, cls.data_target_lbl, delta=delta, mask=None, # Don't mask labels - use undropped labels for training 
                                        func=lambda x: np.array(x) == 1) # is it a set
            label_i = raw_label_i > 0.5
            _, raw_label_i = lin_interp(obj, cls.data_target_lbl, delta=delta,  
                                        mask=None, # Don't mask labels - use undropped labels for training 
                                        func=lambda x: np.array(x) == 1) # is it a set
            label_i = raw_label_i > 0.5

            _, raw_defined_i = lin_interp(obj, 'is_defined', delta=delta, 
                                          mask=None, # Don't mask labels - use undropped labels for training 
                                          func=lambda x: np.array(x) == 1) # is it a set
            defined_i = raw_defined_i > 0.5
        #
        return t, xi, y, label_i, defined_i

