from LeWagon_FinalProject.data import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from hilbert import decode, encode
import plotly.graph_objects as go
import bezier
from matplotlib.collections import LineCollection

from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, MultiLine, Plot, Scatter, HoverTool
from bokeh.plotting import figure, output_file, save, output_notebook
from datashader.bundling import hammer_bundle

import holoviews as hv
from holoviews import opts

from holoviews.operation.datashader import datashade, dynspread
from holoviews.operation import decimate

from dask.distributed import Client

class Grapho():

    def __init__(self, raw_data_similar, raw_topics):
        """
            df: pandas DataFrame with id1, id2, weight and topic from BERTopic
        """
        self.raw_data_similar = raw_data_similar
        self.raw_topics = raw_topics

    def prepare_data(self):
        self.raw_topics.rename(columns={'document1': 'id1'}, inplace=True)
        self.raw_topics.rename(columns={'document2': 'id2'}, inplace=True)

        self.raw_topics.id1 = self.raw_topics.id1.astype('int32')
        self.raw_topics.id2 = self.raw_topics.id2.astype('int32')

        np.fill_diagonal(self.raw_data_similar, 0)
        self.raw_data_similar = pd.DataFrame(self.raw_data_similar)
        self.raw_data_similar = self.raw_data_similar.stack().reset_index()
        self.raw_data_similar.columns = ['id1', 'id2', 'weight']
        # self.raw_data_similar.id1 = self.raw_data_similar.id1.astype(str)
        # self.raw_data_similar.id2 = self.raw_data_similar.id2.astype(str)


    def filter(self, weight=0.3, topic_1=True, min_topic_news=0):
        x = self.raw_topics.groupby('topic').count().reset_index()
        x = x.loc[x.id1 >= min_topic_news].topic.values
        self.topics = self.raw_topics.loc[self.raw_topics["topic"].isin(x)]
        values = self.topics.id1.unique()
        self.similar = self.raw_data_similar.loc[(self.raw_data_similar["id1"].isin(values)) & (self.raw_data_similar["id2"].isin(values))]
        self.similar = self.similar.loc[(self.similar['weight'] > weight) & (self.similar['id1'] != self.similar['id2'])]
        values = np.concatenate((self.similar.id1.unique(), self.similar.id2.unique()))
        self.topics = self.topics.loc[self.topics["id1"].isin(values)]
        if topic_1 == False:
            values = self.similar.loc[self.similar.topic == -1].id1.unique()
            self.similar = self.similar.loc[self.similar.topic != -1]
            self.similar = self.similar[np.logical_not(self.similar["id2"].isin(list(values)))]

    def definition_size(self, bins=7, labels=[1,1.5,2,2.5,3,3.5,4]):
        self.bins = bins
        self.labels = labels
        X = self.similar.groupby(['id1']).sum().rename(columns={'weight':'sum_weight'})

        X['size'] = pd.cut(X['sum_weight'], bins=self.bins, labels=self.labels)

        self.similar = self.similar.merge(X[['size']],left_on='id1',right_index=True)

    def generate_graph(self, tam=True):
        self.G = nx.Graph()
        self.size = tam
        for ind in self.similar.id1.unique():
            no_1 = ind
            if self.size:
                size =self.similar['size'][ind]
                self.G.add_node(no_1,size =size,color = cor)#, ref = ref)
        for ind in self.similar.index:
            no_1 = self.similar['id1'][ind]
            no_2 = self.similar['id2'][ind]
            weight = self.similar['weight'][ind]
            cor = self.similar['topic'][ind]
            self.G.add_edge(no_1, no_2, weight = weight)

    def calculate_order(self, strategy='topic_similarity', remove_1=True, tam=True):
        if strategy == 'louvain':
            self.generate_graph(tam)
            partition = community_louvain.best_partition(self.G)
            t = pd.DataFrame.from_dict(partition,orient='index').reset_index()
            t = t.rename(columns = {'index':'node',0:'center'})
            t['node_2'] = t['node']
            self.order_values = t.groupby('center').agg({'node': 'count', 'node_2': list}).sort_values('node', ascending=False)
            position = self.order_values['node_2'].sum()
            position = pd.DataFrame(np.array(position), columns=['id1']).reset_index()
            self.similar = self.similar.merge(position, on='id1').sort_values('index').drop(columns='index')
        elif strategy == 'important_topic':
            self.similar['id1_2'] = self.similar['id1']
            __filter_links = self.similar.drop_duplicates('id1').groupby('topic').agg({'id1': 'count', 'id1_2': list}).sort_values('id1', ascending=False)
            if remove_1:
                df_first = __filter_links.head(1)
                df_rest = __filter_links.drop(index=-1)
                self.order_values = pd.concat([df_rest, df_first], ignore_index=True)
            else:
                self.order_values = __filter_links
            position = self.order_values['id1_2'].sum()
            position = pd.DataFrame(np.array(position), columns=['id1']).reset_index()
            self.similar = self.similar.merge(position, on='id1').sort_values('index').drop(columns=['index', 'id1_2'])
        elif strategy == 'topic_center':
            self.similar['id1_2'] = self.similar['id1']
            __filter_links = self.similar.drop_duplicates('id1').groupby('topic').agg({'id1': 'count', 'id1_2': list}).sort_values('id1', ascending=False)
            df_first = __filter_links.head(1).copy()
            if df_first.index == -1:
                df_even = __filter_links.iloc[::2].copy().drop(index=-1).sort_values(by='id1', ascending=True)
                df_odd = __filter_links.iloc[1::2]  # odd
                if df_odd.shape[0] > df_even.shape[0]:
                    df_even = pd.concat([df_first, df_even], ignore_index=True)
                    self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
                elif df_odd.shape[0] < df_even.shape[0]:
                    df_odd = pd.concat([df_odd, df_first], ignore_index=True)
                    self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
                else:
                    if df_odd.iloc[0].id1 >= df_even.iloc[-1].id1:
                        df_odd = pd.concat([df_odd, df_first], ignore_index=True)
                        self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
                    else:
                        df_even = pd.concat([df_first, df_even], ignore_index=True)
                        self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
            else:
                df_even = __filter_links.iloc[::2].copy().sort_values(by='id1', ascending=True).reset_index()
                df_odd = __filter_links.iloc[1::2].reset_index()  # odd
                if df_odd.shape[0] > df_even.shape[0]:
                    self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
                elif df_odd.shape[0] < df_even.shape[0]:
                    self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
                else:
                    if df_odd.iloc[0].id1 >= df_even.iloc[-1].id1:
                        self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
                    else:
                        self.order_values = pd.concat([df_even, df_odd], ignore_index=True)
            position = self.order_values['id1_2'].sum()
            position = pd.DataFrame(np.array(position), columns=['id1']).reset_index()
            self.similar = self.similar.merge(position, on='id1').sort_values('index').drop(columns=['index', 'id1_2'])
        elif strategy == 'topic_similarity':
            self.list_topics = pd.DataFrame(self.topics.topic.diff())
            self.list_topics.iloc[0] = 0
            return

        list_topics = []
        for idx, row in self.order_values.iterrows():
            list_topics += [row['topic']]*row['id1']
        self.list_topics = pd.DataFrame(np.append(np.array([0]), np.diff(np.array(list_topics))))

    def hilbert_d(self, num_bits, num_dims):
        self.num_bits = num_bits
        self.num_dims = num_dims
        # The maximum Hilbert integer.
        max_h = 2**(num_bits*num_dims)

        # Generate a sequence of Hilbert integers.
        hilberts = np.arange(max_h)

        # Compute the 2-dimensional locations.
        locs = decode(hilberts, num_dims, num_bits)
        x = np.array(locs[:,0])
        y = np.array(locs[:,1])
        if num_dims == 3:
            z = np.array(locs[:,2])
            df = pd.DataFrame(np.array([x,y, z]).T, columns=['x', 'y', 'z'])
        else:
            df = pd.DataFrame(np.array([x,y]).T, columns=['x', 'y'])
        self.hilbert_graph = df

    def plot_hilbert(self, data=False):
        fig = plt.figure(figsize=(16,4))
        if self.num_dims == 3:
            ax = fig.add_subplot(1,1,1, projection='3d')
            ax.plot(self.hilbert_graph['x'],self.hilbert_graph['y'],self.hilbert_graph['z'], '.-')
            if data:
                ax.scatter(self.ponto['x'],self.ponto['y'],self.ponto['z'])
            ax.set_title('%d bits per dimension' % (self.num_bits))
            ax.set_xlabel('dim 1')
            ax.set_ylabel('dim 2')
            ax.set_zlabel('dim 3')

        else:
            ax = fig.add_subplot(1,1,1)
            ax.plot(self.hilbert_graph['x'],self.hilbert_graph['y'], '.-')
            if data:
                ax.scatter(self.ponto['x'],self.ponto['y'])
            ax.set_title('%d bits per dimension' % (self.num_bits))
            ax.set_xlabel('dim 1')
            ax.set_ylabel('dim 2')


    def calculate_spacing(self, spacing_mul, strategy):
        if strategy=='ia':
            unique_filter = self.similar['id1'].unique()
            spacing_2 = (self.hilbert_graph.shape[0]-1)/len(unique_filter) * len(unique_filter)
            unique_topics = self.topics.topic.unique()
            spacing = (self.hilbert_graph.shape[0]-1)/(len(self.similar['id1'].unique()) + spacing_mul* (len(unique_topics)-1))
            self.spacing = pd.DataFrame(np.array([spacing] * len(unique_filter)))
            self.spacing.index = self.topics.index
            for idx, topic in enumerate(unique_topics):
                x = self.topics.loc[self.topics.topic == topic]
                x['similarity_previous_document'].iloc[0] = 0
                value = x.sum().similarity_previous_document
                total = (self.spacing.loc[x.index].sum()-self.spacing.loc[x.index[0]])[0]
                document_distance = x.similarity_previous_document *(total/value)
                self.spacing.loc[document_distance.index, 0] = document_distance
            dif = (spacing_2 - self.spacing.sum())[0]
            topic_distance = self.topics.similarity_previous_topic.unique()
            topic_distance = np.delete(topic_distance, 0)
            topic_distance = (topic_distance * dif/topic_distance.sum())
            self.spacing[0].loc[self.list_topics.loc[self.list_topics.topic != 0].index] = topic_distance
        else:
            unique_filter = self.similar['id1'].unique()
            spacing_2 = (self.hilbert_graph.shape[0]-1)/len(unique_filter) * len(unique_filter)
            if spacing_mul == 1:
                spacing = (self.hilbert_graph.shape[0]-1)/len(self.similar['id1'].unique())
                self.spacing = pd.DataFrame(np.array([spacing] * len(self.similar['id1'].unique())))
            else:
                unique_topics = self.topics.topic.unique()
                spacing = (self.hilbert_graph.shape[0]-1)/(len(self.similar['id1'].unique()) + spacing_mul* (len(unique_topics)-1))
                self.spacing = pd.DataFrame(np.array([spacing] * len(self.similar['id1'].unique())))
                dif = (spacing_2 - self.spacing.sum()) / (len(unique_topics)-1) + spacing
                self.spacing.loc[self.list_topics.loc[self.list_topics[0] != 0].index] = dif[0]


    def calculate_hilbert_line(self, spacing_mul=1, strategy=None):

        dif_df = self.hilbert_graph.diff()
        self.calculate_spacing(spacing_mul, strategy=strategy)
        ponto = []
        idx = 1
        increment = 0
        space = 0
        for i, value in enumerate(self.spacing.index):
            space = space + self.spacing[0].iloc[i]
            division = space // 1
            if division > 0:
                idx = idx + division
                increment = space % 1
                space = increment
                space_mul = 0
            if idx <= self.hilbert_graph.shape[0]-1:
                x_dif = dif_df['x'][idx]
                y_dif = dif_df['y'][idx]
                if x_dif == 0:
                    ponto_x = self.hilbert_graph['x'][idx-1]
                else:
                    if dif_df['x'][idx] == 1:
                        ponto_x = self.hilbert_graph['x'][idx-1] + space
                    else:
                        ponto_x = self.hilbert_graph['x'][idx-1] - space
                if y_dif == 0:
                    ponto_y = self.hilbert_graph['y'][idx-1]
                else:
                    if dif_df['y'][idx] > 0:
                        ponto_y = self.hilbert_graph['y'][idx-1] + space
                    else:
                        ponto_y = self.hilbert_graph['y'][idx-1] - space
                if self.num_dims == 3:
                    z_dif = dif_df['z'][idx]
                    if z_dif == 0:
                        ponto_z = self.hilbert_graph['z'][idx-1]
                    else:
                        if dif_df['z'][idx] == 1:
                            ponto_z = self.hilbert_graph['z'][idx-1] + space
                        else:
                            ponto_z = self.hilbert_graph['z'][idx-1] - space
                    ponto.append([ponto_x, ponto_y, ponto_z])
                else:
                    ponto.append([ponto_x, ponto_y])

        if self.num_dims == 2:
            ponto.append([self.hilbert_graph['x'].iloc[-1],self.hilbert_graph['y'].iloc[-1]])
        else:
            ponto.append([self.hilbert_graph['x'].iloc[-1], self.hilbert_graph['y'].iloc[-1], self.hilbert_graph['z'].iloc[-1]])
        a = np.array(ponto).T[0]
        b = np.array(ponto).T[1]
        if self.num_dims == 3:
            c = np.array(ponto).T[2]
            values = np.array([a, b, c]).T
            columns = ['x', 'y', 'z']
            coord = np.stack((np.array(a), np.array(b), np.array(c)), axis=-1)
        else:
            values = np.array([a, b]).T
            columns = ['x', 'y']
            coord = np.stack((np.array(a), np.array(b)), axis=-1)
        self.ponto = pd.DataFrame(values, columns=columns)
        self.dict_for_position(coord.tolist())


    def calculate_hilbert_line_old(self, spacing_mul=1, strategy=None):

        dif_df = self.hilbert_graph.diff()
        self.calculate_spacing(spacing_mul, strategy=strategy)
        ponto = []
        idx = 1
        space_mul = 0
        increment = 0
        last_space = self.spacing[0].iloc[0]
        for i in range(len(self.similar['id1'].unique())):
            space_mul +=1
            if last_space != self.spacing[0].iloc[i]:
                space = (1 * self.spacing[0].iloc[i]) + increment
            else:
                space = (space_mul * self.spacing[0].iloc[i]) + increment
            last_space = self.spacing[0].iloc[i]
            division = space // 1
            if division > 0:
                idx = idx + division
                increment = space % 1
                space = increment
                space_mul = 0
            if idx <= self.hilbert_graph.shape[0]-1:
                x_dif = dif_df['x'][idx]
                y_dif = dif_df['y'][idx]
                if x_dif == 0:
                    ponto_x = self.hilbert_graph['x'][idx-1]
                else:
                    if dif_df['x'][idx] == 1:
                        ponto_x = self.hilbert_graph['x'][idx-1] + space
                    else:
                        ponto_x = self.hilbert_graph['x'][idx-1] - space
                if y_dif == 0:
                    ponto_y = self.hilbert_graph['y'][idx-1]
                else:
                    if dif_df['y'][idx] > 0:
                        ponto_y = self.hilbert_graph['y'][idx-1] + space
                    else:
                        ponto_y = self.hilbert_graph['y'][idx-1] - space
                if self.num_dims == 3:
                    z_dif = dif_df['z'][idx]
                    if z_dif == 0:
                        ponto_z = self.hilbert_graph['z'][idx-1]
                    else:
                        if dif_df['z'][idx] == 1:
                            ponto_z = self.hilbert_graph['z'][idx-1] + space
                        else:
                            ponto_z = self.hilbert_graph['z'][idx-1] - space
                    ponto.append([ponto_x, ponto_y, ponto_z])
                else:
                    ponto.append([ponto_x, ponto_y])

        if self.num_dims == 2:
            ponto.append([self.hilbert_graph['x'].iloc[-1],self.hilbert_graph['y'].iloc[-1]])
        else:
            ponto.append([self.hilbert_graph['x'].iloc[-1], self.hilbert_graph['y'].iloc[-1], self.hilbert_graph['z'].iloc[-1]])
        a = np.array(ponto).T[0]
        b = np.array(ponto).T[1]
        if self.num_dims == 3:
            c = np.array(ponto).T[2]
            values = np.array([a, b, c]).T
            columns = ['x', 'y', 'z']
            coord = np.stack((np.array(a), np.array(b), np.array(c)), axis=-1)
        else:
            values = np.array([a, b]).T
            columns = ['x', 'y']
            coord = np.stack((np.array(a), np.array(b)), axis=-1)
        self.ponto = pd.DataFrame(values, columns=columns)
        self.dict_for_position(coord.tolist())

    def add_list_edge_node(self):
        self.G = nx.Graph()
        new_similar = self.similar[self.similar.weight>0.7]
        self.G.add_edges_from(list(zip(new_similar.id1, new_similar.id2)))
        self.G.add_nodes_from(self.pos)

    def dict_for_position(self, values):
        self.pos = {}
        x = []
        y = []
        keys = self.topics['id1'].unique()
        for i in range(len(keys)):
            self.pos[keys[i]] = values[i]
            x.append(values[i][0])
            y.append(values[i][1])
        self.topics['x'] = x
        self.topics['y'] = y

    def curved_edges(self, dist_ratio=0.2, bezier_precision=20, polarity='random'):
        # Get nodes into np array
        edges = np.array(self.G.edges())
        l = edges.shape[0]

        if polarity == 'random':
            # Random polarity of curve
            rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
        else:
            # Create a fixed (hashed) polarity column in the case we use fixed polarity
            # This is useful, e.g., for animations
            rnd = np.where(np.mod(np.vectorize(hash)(edges[:,0])+np.vectorize(hash)(edges[:,1]),2)==0,-1,1)

        # Coordinates (x,y) of both nodes for each edge
        # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
        # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
        u, inv = np.unique(edges, return_inverse = True)
        coords = np.array([self.pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
        coords_node1 = coords[:,0,:]
        coords_node2 = coords[:,1,:]

        # Swap node1/node2 allocations to make sure the directionality works correctly
        should_swap = coords_node1[:,0] > coords_node2[:,0]
        coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]

        # Distance for control points
        dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

        # Gradients of line connecting node & perpendicular
        m1 = (coords_node2[:,1]-coords_node1[:,1])/(coords_node2[:,0]-coords_node1[:,0])
        m2 = -1/m1

        # Temporary points along the line which connects two nodes
        # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
        t1 = dist/np.sqrt(1+m1**2)
        v1 = np.array([np.ones(l),m1])
        coords_node1_displace = coords_node1 + (v1*t1).T
        coords_node2_displace = coords_node2 - (v1*t1).T

        # Control points, same distance but along perpendicular line
        # rnd gives the 'polarity' to determine which side of the line the curve should arc
        t2 = dist/np.sqrt(1+m2**2)
        v2 = np.array([np.ones(len(edges)),m2])
        coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
        coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

        # Combine all these four (x,y) columns into a 'node matrix'
        node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

        # Create the Bezier curves and store them in a list
        curveplots = []
        for i in range(l):
            nodes = node_matrix[:,i,:].T
            curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0,1,bezier_precision)).T)

        # Return an array of these curves
        curves = np.array(curveplots)
        return curves


    def plot_bezier(self, tam=50):

        self.curves = self.curved_edges()
        # lc = LineCollection(curves, color='b', alpha=0.02)
        lc = LineCollection(self.curves, color='silver', alpha=0.05,linewidths=3.5)
        # Plot
        self.fig = plt.figure(figsize=(20,20))
        self.fig = plt.figure(figsize=(900/96, 900/96), dpi=96)
        #print(plt.rcParams["figure.figsize"])
        nx.draw_networkx_nodes(self.G, self.pos, node_size=((1-(1/self.topics.document1_size.values))*tam)+1, node_color='b', alpha=0.4)
        plt.gca().add_collection(lc)
        nx.draw_networkx_nodes(self.G, self.pos, node_size=((1-(1/self.topics.document1_size.values))*tam)+1, node_color='b', alpha=0.4)
        # plt.grid(b=None)
        plt.style.use("dark_background")
        plt.show(block=True)

    def curve_to_df(self):
        self.curves = self.curved_edges()
        self.curve_x = []
        self.curve_y = []
        for idx, line in enumerate(self.curves):
            line2 = line.T
            self.curve_x.append(line2[0])
            self.curve_y.append(line2[1])

    def topic_minmax(self):
        self.topic_limits = self.topics[['topic', 'x', 'y']].groupby('topic').agg({"x": [np.min, np.max, np.max, np.min, np.min], "y": [np.min, np.min, np.max, np.max, np.min]})
        self.topic_limits.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3', 'y4', 'y5']
        self.topic_limits['x']= self.topic_limits[['x1', 'x2', 'x3', 'x4', 'x5']].values.tolist()
        self.topic_limits['y']= self.topic_limits[['y1', 'y2', 'y3', 'y4', 'y5']].values.tolist()
        self.topic_limits = self.topic_limits[['x', 'y']].reset_index()


    def change_sentiment(self):
        self.topics.sentimet_classification = self.topics.sentimet_classification.map({'neutral': 'grey', 'positive':'gold' , 'negative': 'red'})
    

    def plot_bokeh(self, limits=False, save_file=False, output_filename='plot.html', output_note=True, sentiment=False, title=False):

        if output_note:
            output_notebook()

        source = ColumnDataSource(dict(
                xs=self.curve_x,
                ys=self.curve_y,
            )
        )

        if title:
            title_values = self.topics['document1_title'].values
        else:
            title_values = self.topics['topic_name'].values
        x = self.topics.document1_size.values
        
        #Lucas
        mx = 16
        mn = 6
        newvalue= (mx-mn)/(max(x)-min(x))*(x-max(x))+mx
        
        
        
        
        source2 = ColumnDataSource(dict(
                x=self.topics['x'].values,
                y=self.topics['y'].values,
                size = newvalue,
                #size = ((1-(1/self.topics.document1_size.values))*5.5)+1,
                #size = self.topics['document1_size'].values / 15,
                topic = self.topics['topic_name'].values,
                sentiment = self.topics['sentimet_classification'].values,
                
                title = title_values,

            )
        )
        #Lucas
        from bokeh.transform import linear_cmap
        from colorcet import CET_D3
        from colorcet import fire
        
        y = self.raw_topics['setiment_scale']
        mapper = linear_cmap(field_name='y', palette=fire ,low=min(y) ,high=max(y))
        
        curdoc().theme = 'dark_minimal'
        plot = figure(plot_width=900, plot_height=900)
        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None
        plot.axis.visible = False
        
        
        glyph = MultiLine(xs="xs", ys="ys", line_width=1.5, line_alpha=0.1, line_color='#D4AF37')

        hover_off = plot.add_glyph(source, glyph)

        if limits:
            source3 = ColumnDataSource(dict(
                    x=list(np.array(list(self.topic_limits.x.values))),
                    y=list(np.array(list(self.topic_limits.y.values))),
                    topic = self.topic_limits['topic'].values,
                )
            )

            glyph = MultiLine(xs="x", ys="y", line_width=1, line_alpha=0.6, line_color='#ff3300')
            hover_off_2 = plot.add_glyph(source3, glyph)

        if sentiment:
            glyph = Scatter(x="x", y="y", size='size', line_color='#996515', fill_color=mapper)
        else:
            glyph = Scatter(x="x", y="y", size='size', line_color='#996515', fill_color='#996515')

        hover_on = plot.add_glyph(source2, glyph)

        if title:
            tooltips = [
                #("index", "$index"),
                #("(x,y)", "($x, $y)"),
                ("topic", "@topic"),
                ("title", "@title")
            ]
        else:
            tooltips = [
                #("index", "$index"),
                #("(x,y)", "($x, $y)"),
                ("topic", "@topic"),
                ("title", "@title")
            ]

        plot.add_tools(HoverTool(tooltips=tooltips, renderers=[hover_on]))
        
        curdoc().add_root(plot)

        if save_file:
            output_file(output_filename)
            save(plot)

        show(plot, notebook_handle=True)
        
    def define_hb(self):

        ds_nodes = pd.DataFrame(self.pos).transpose().reset_index()
        ds_nodes.columns = ['name', 'x', 'y']
        ds_edges = self.similar[['id1', 'id2']].reset_index(drop=True)
        ds_edges.columns = ['source', 'target']
        ds_edges['source'] = ds_edges['source'].astype('int64')
        ds_edges['target'] = ds_edges['target'].astype('int64')
        ds_nodes['name'] = ds_nodes['name'].astype('int64')
        self.hb = hammer_bundle(ds_nodes, ds_edges)

    def plot_hb(self):
        client = Client()

        hv.notebook_extension('bokeh','matplotlib')

        decimate.max_samples=20000
        dynspread.threshold=0.01
        sz = dict(width=150,height=150)

        opts.defaults(
            opts.RGB(width=400, height=400, xaxis=None, yaxis=None, show_grid=False, bgcolor="black"))


        self.hb.plot(x="x", y="y", figsize=(9,9))

    def make_edge(self, x, y, text, width):

        '''Creates a scatter trace for the edge between x's and y's with given width

        Parameters
        ----------
        x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])

        y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])

        width: the width of the line

        Returns
        -------
        An edge trace that goes between x0 and x1 with specified width.
        '''
        return  go.Scatter(x = x,
                           y = y,
                           line = dict(width = width,
                                       color = 'cornflowerblue'),
                           hoverinfo = 'text',
                           text = ([text]),
                           mode = 'lines')


    def add_edge(self):
        # For each edge, make an edge_trace, append to list
        self.edge_trace = []
        for edge in self.G.edges():

            if self.G.edges()[edge]['weight'] > 0:
                char_1 = str(edge[0])
                char_2 = str(edge[1])

                x0, y0 = self.pos[char_1]
                x1, y1 = self.pos[char_2]

                #text   = char_1 + '--' + char_2 + ': ' + str(G.edges()[edge]['weight'])
                text = ''

                trace  = self.make_edge([x0, x1, None], [y0, y1, None], text,
                                   0.3*self.G.edges()[edge]['weight']**1.75)

                self.edge_trace.append(trace)

    def make_node_trace(self):

        # Make a node trace
        self.node_trace = go.Scatter(x         = [],
                                y         = [],
                                text      = [],
                                textposition = "top center",
                                textfont_size = 10,
                                mode      = 'markers+text',
                                #hoverinfo = 'none',
                                marker    = dict(color = [],
                                                 size  = [],
                                                 line  = None))
        # For each node in midsummer, get the position and size and add to the node_trace
        #node_trace['marker']['color']=newcolors
        for node in self.G.nodes():
            x, y = self.pos[str(node)]
            self.node_trace['x'] += tuple([x])
            self.node_trace['y'] += tuple([y])
            if self.size:
                self.node_trace['marker']['size'] += tuple([5*self.G.nodes()[node]['size']])
            self.node_trace['text'] #+= tuple(['<b>' + node + '</b>'])

    def plot_graphos(self):

        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = go.Figure(layout = layout)

        for trace in self.edge_trace:
            fig.add_trace(trace)

        fig.add_trace(self.node_trace)

        fig.update_layout(showlegend = False)

        fig.update_xaxes(showticklabels = False)

        fig.update_yaxes(showticklabels = False)

        fig.show()

