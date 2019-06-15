from .base_model import BaseModel

class SingleTrackModel(BaseModel):

    def create_features_and_times(self, data, angle=77):
        t, xi, y, label_i, defined_i = self.build_features(data, skip_label=True)
        min_ndx = 0
        max_ndx = len(t) - self.time_points
        features = []
        for i in range(min_ndx, max_ndx):
            raw_features = y[i:i+(self.time_points+1)]
            features.append(self.cook_features(raw_features, angle=angle, noise=0)[0])
        times = t[self.time_points//2:-self.time_points//2]
        return features, times

