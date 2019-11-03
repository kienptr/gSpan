import codecs
import collections
import itertools
import time
import copy
import pandas as pd


VACANT_EDGE_ID = -1
VACANT_VERTEX_ID = -1
VACANT_EDGE_LABEL = -1
VACANT_VERTEX_LABEL = -1
VACANT_GRAPH_ID = -1
AUTO_EDGE_ID = -1


def record_timestamp(func):
	"""Record timestamp before and after call of `func`."""
	def deco(self):
		self.timestamps[func.__name__ + '_in'] = time.time()
		func(self)
		self.timestamps[func.__name__ + '_out'] = time.time()
	return deco


class Edge(object):
		"""Lớp biểu diễn cạnh của đồ thị."""
		
		def __init__(self, id, frm, to, label):						
			self.id = id
			self.frm = frm
			self.to = to
			self.label = label

				
class Vertex(object): 
	"""Lớp biểu diễn các đỉnh của đồ thị."""
	
	def __init__(self, id, label):
		"""Khởi tạo một đỉnh """
		self.id = id
		self.label = label
		self.edges = dict() 	# Tập các cạnh chứa đỉnh
		self.frequent_1edge_subgraphs = dict()

	def add_edge(self, eid, frm, to, elb):
		"""Thêm cạnh chứa đỉnh."""
		self.edges[to] = Edge(eid, frm, to, elb)
	

class Graph(object):
	"""Lớp biểu diễn đồ thị."""
	
	def __init__(self, id):		
		self.id = id
		self.vertices = dict()	# Tập các đỉnh
		self.edges = dict()		# Tập cách cạnh
		self.eid_auto_increment = True
		self.counter = itertools.count()
	
	def add_vertex(self, vid, vlabel):
		""" Thêm 1 đỉnh vào đồ thị. """
		if vid in self.vertices:
			return self
		self.vertices[vid] = Vertex(vid, vlabel)
		return self

	def add_edge(self, eid, frm, to, elb):
		""" Thêm 1 cạnh vào đồ thị. """
		if (frm is self.vertices and to in self.vertices and to in self.vertices[frm].edges):
			return self
		if self.eid_auto_increment:
			eid = next(self.counter)     
		self.vertices[frm].add_edge(eid, frm, to, elb)        
		self.vertices[to].add_edge(eid, to, frm, elb) 
		return self               

	def display(self):
		"""Hiển thị đồ thị dạng text."""
		display_str = ''
		print('t # {}'.format(self.id))
		for vid in self.vertices:
			print('v {} {}'.format(vid, self.vertices[vid].label))
			display_str += 'v {} {} '.format(vid, self.vertices[vid].label)
		for frm in self.vertices:
			edges = self.vertices[frm].edges
			for to in edges:				
				if frm < to:
					print('e {} {} {}'.format(frm, to, edges[to].label))
					display_str += 'e {} {} {} '.format(
						frm, to, edges[to].label)
				
		return display_str

	def plot(self):
		"""Vẽ đồ thị"""
		try:
			import networkx as nx
			import matplotlib.pyplot as plt
		except Exception as e:
			print('Can not plot graph: {}'.format(e))
			return
		gnx = nx.Graph() 
		vlbs = {vid: v.label for vid, v in self.vertices.items()}
		elbs = {}
		for vid, v in self.vertices.items():
			gnx.add_node(vid, label=v.label)
		for vid, v in self.vertices.items():
			for to, e in v.edges.items():
				if vid < to:
					gnx.add_edge(vid, to, label=e.label)
					elbs[(vid, to)] = e.label
		fsize = (min(16, 1 * len(self.vertices)),
				 min(16, 1 * len(self.vertices)))
		# plt.figure(3, figsize=fsize)		
		if(len(self.vertices)) > 2:
			pos = nx.spectral_layout(gnx)
		else:
			pos = nx.spring_layout(gnx)
		nx.draw_networkx(gnx, pos, with_labels=True, labels=vlbs)
		nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)
		plt.show()


class DFSEdge(object):
	"""Biểu diễn cạnh theo mã DFS"""
		
	def __init__(self, frm,to, vevLabel):
		self.frm = frm
		self.to = to
		self.vevLabel = vevLabel	# bộ 3 nhãn 2 đỉnh và nhãn của cạnh
	
	def display(self):
		frm, to, (vlb1, elb, vlb2) = self.frm, self.to, self.vevLabel
		print('({}, {}, {}, {}, {})'.format(frm, to, vlb1, elb, vlb2))

	def __eq__(self, other):
		"""Kiểm tra 2 cạnh bằng nhau."""
		return (self.frm == other.frm and
				self.to == other.to and
				self.vevLabel == other.vevLabel)

	def __lt__(self, other):
		"""Kiểm tra cạnh nhỏ hơn cạnh khác."""
		if self.frm == other.frm and self.to == other.to:	# 2 cạnh cùng thứ tự ==> so sánh các nhãn
			return self.vevLabel < other.vevLabel  
		elif self.frm < self.to and other.frm < other.to:		# 2 cạnh đều là forward
			return self.to < other.to or (self.to == other.to and self.frm > other.frm)
		elif self.frm > self.to and other.frm > other.to:		# 2 cạnh đều là backward
			return self.frm < other.frm or(self.frm == other.frm and self.to < other.to)
		elif self.frm < self.to and other.frm > other.to:		# cạnh 1 là forward cạnh 2 là backward
			return self.to < other.frm
		else: # cạnh 1 là backward cạnh 2 là forward
			return self.frm < other.to
		return False

	def __ne__(self, other):
		"""Kiểm tra 2 cạnh không bằng nhau"""
		return not self.__eq__(other)

	def __repr__(self):
		"""Biểu diễn dạng text"""
		return '(frm={}, to={}, vevlb={})'.format(
			self.frm, self.to, self.vevLabel
		)


class DFSCode(list):
	"""DFS Code của đồ thị là danh sách các cạnh (DFSEdges)"""
	
	def __init__(self):
		self.rmp = list()	# đường dẫn bên phải nhất (rightmost path)
	
	def __eq__(self, other):
		"""Kiểm tra 2 DFScode có bằng nhau không"""
		la, lb = len(self), len(other)
		if la != lb:
			return False
		for i in range(la):
			if self[i] != other[i]:
				return False
		return True

	def __ne__(self, other):
		"""Kiểm tra 2 DFScode không bằng nhau."""
		return not self.__eq__(other)

	def __lt__(self, other):
		"""Kiểm tra DFScode có nhỏ hơn DFSCode khác không."""
		if len(self) != len(other):
			raise Exception('Khích thước 2 đồ thị không bằng nhau!')
		for i in range(0, len(self)):
			if self[i] < other[i]:
				return True			
		return False

	def __repr__(self):
		"""Biểu diễn DFScode dạng text."""
		return ''.join(['[', ','.join(
			[str(dfsedge) for dfsedge in self]), ']']
		)

	def to_graph(self, gid=-1, is_undirected=True):
		"""Construct a graph according to the dfs code."""
		g = Graph(gid)                  
		for dfsedge in self:
			frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevLabel
			if vlb1 != VACANT_VERTEX_LABEL:
				g.add_vertex(frm, vlb1)
			if vlb2 != VACANT_VERTEX_LABEL:
				g.add_vertex(to, vlb2)
			g.add_edge(AUTO_EDGE_ID, frm, to, elb)
		return g	

	def build_rmpath(self):
		"""Lấy đường dẫn bên phải."""
		self.rmpath = list()
		old_frm = None
		for i in range(len(self) - 1, -1, -1):
			dfsedge = self[i]
			frm, to = dfsedge.frm, dfsedge.to
			if frm < to and (old_frm is None or to == old_frm):
				self.rmpath.append(i)
				old_frm = frm
		return self

	def get_num_vertices(self):
		"""Đếm số đỉnh trong đồ thị."""
		return len(set(
			[dfsedge.frm for dfsedge in self] +
			[dfsedge.to for dfsedge in self]
		))
	
	def plot(self):    	
		g = self.to_graph(gid=-1)        
		g.plot()		

	def display(self):
		for dfsedge in self:
			dfsedge.display()


class PDFS(object):
	"""Lớp PDFS cho biết cạnh thuộc đồ thị nào (hình chiếu của cạnh)."""

	def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):		
		self.gid = gid
		self.edge = edge
		self.prev = prev

	def __repr__(self):
		""" in ra dạng text"""
		return '(gid = {}, edge = ({}, {}, {}))'.format(self.gid, self.edge.frm, self.edge.to, self.edge.label)

	def display(self):
		""" in ra dạng text"""
		print('gid = {}, edge = ({}, {}, {})'.format(self.gid, self.edge.frm, self.edge.to, self.edge.label))

class Projected(list):
	"""Danh sách các PDFS của một đồ thị."""
	
	def __init__(self):		
		super(Projected, self).__init__()

	def push_back(self, gid, edge, prev):		
		self.append(PDFS(gid, edge, prev))
		return self

	def __repr__(self):
		"""Biểu diễn dạng text."""
		return ''.join(['[', ','.join(
			[str(pdfs) for pdfs in self]), ']']
		)

class History(object):
	"""Lưu các cạnh đã được duyệt."""

	def __init__(self, g, pdfs):		
		super(History, self).__init__()
		self.edges = list()
		self.vertices_used = collections.defaultdict(int)
		self.edges_used = collections.defaultdict(int)
		if pdfs is None:
			return
		while pdfs:
			e = pdfs.edge
			self.edges.append(e)
			(self.vertices_used[e.frm],
				self.vertices_used[e.to],
				self.edges_used[e.id]) = 1, 1, 1

			pdfs = pdfs.prev		# Cạnh vừa được duyệt trước
		self.edges = self.edges[::-1]

	def has_vertex(self, vid):
		"""Kiểm tra đỉnh đã được duyệt chưa."""
		return self.vertices_used[vid] == 1

	def has_edge(self, eid):
		"""Kiểm tra cạnh đã được duyệt chưa."""
		return self.edges_used[eid] == 1


class gSpan(object):
	"""Giải thuật gSpan"""
	
	def __init__(self, database_file_name, min_support=10):		
		self.fileName = database_file_name
		self.minSup = min_support
		self.graphs = dict()
		self.frequent_1edge_subgraphs = list()
		self.timestamps = dict()
		# self.min_num_vertices = min_num_vertices
		# self.max_num_vertices = max_num_vertices
		self.DFScode = DFSCode()
		self.support = 0
		self.frequent_subgraphs = list()
		self.counter = itertools.count()
		self.report_df = pd.DataFrame()
	
	def time_stats(self):
		"""Print stats of time."""
		func_names = ['readGraphs', 'get_frequent_1edge_subgraphs', 'run']
		time_deltas = collections.defaultdict(float)
		for fn in func_names:
			time_deltas[fn] = round(
				self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
				2
			)
		print('Read:\t{} s'.format(time_deltas['readGraphs']))
		print('Gen 1edge subgraph:\t{} s'.format(time_deltas['get_frequent_1edge_subgraphs']))
		print('Mine:\t{} s'.format(
			time_deltas['run'] - time_deltas['readGraphs']))
		print('Total:\t{} s'.format(time_deltas['run']))
		return self

	@record_timestamp
	def readGraphs(self):
		""" Đọc dữ liệu đồ thị từ file """
		self.graphs = dict()
		with codecs.open(self.fileName, 'r', 'utf-8') as f:
			lines = [line.strip() for line in f.readlines()]  
			temGraph = None          
			graphCounter = 0
			for i, line in enumerate(lines):
				cols = line.split(' ')
				if cols[0] == 't': 		# thêm đồ thị mới
					if temGraph is not None:
						self.graphs[graphCounter] = temGraph
						graphCounter += 1
						temGraph = None
					if cols[-1] == '-1':	# kết thúc	
						break
					temGraph = Graph(graphCounter)                    
				elif cols[0] == 'v':		# thêm đỉnh                	
					temGraph.add_vertex(cols[1], cols[2])
				elif cols[0] == 'e':		# thêm cạnh                	
					temGraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3])
			# nếu file không kết thúc bởi 't # -1'
			if temGraph is not None:
				self.graphs[graphCounter] = temGraph
		return self
	
	@record_timestamp
	def get_frequent_1edge_subgraphs(self):
		"""" lấy các dồ thị con 1 cạnh phổ biến"""
		vlb_counter = collections.Counter()
		vevlb_counter = collections.Counter()
		vlb_counted = set()
		vevlb_counted = set()
		for g in self.graphs.values():			# duyệt từng đồ thị
			for v in g.vertices.values():			# duyệt từng đỉnh
				if (g.id, v.label) not in vlb_counted:	# nếu chưa được duyệt
					vlb_counter[v.label] += 1				# thì tăng bộ đếm
				vlb_counted.add((g.id, v.label))			# đánh dấu đỉnh đã được duyệt
				for to, e in v.edges.items():				# duyệt các cạnh được nối với đỉnh vừa duyệt
					vlb1, vlb2 = v.label, g.vertices[to].label
					if vlb1 < vlb2 or (vlb1 == vlb2 and e.frm < e.to): 
						vevlb_counter[(vlb1, e.label, vlb2)] += 1				# thì tăng bộ đếm
					vevlb_counted.add((g.id, (vlb1, e.label, vlb2)))			# đánh dấu đã đếm 
		# lấy các cạnh phổ biến.
		for (vlb1, elb, vlb2), cnt in vevlb_counter.items():        	
			if cnt >= self.minSup:
				dfsE = DFSEdge(0,1,(vlb1, elb, vlb2))
				dfsC = DFSCode()				
				dfsC.append(dfsE)
				# dfsE.display()    
				# dfsC.plot()  
				# print('support = {}'.format(cnt))                          
				self.frequent_1edge_subgraphs.append(dfsC)  
			else:
				continue     	

	def get_forward_root_edges(self, g, v_frm):
		""" Lấy các cạnh forward từ đỉnh frm"""
		result = []
		# v_frm = g.vertices[frm]		
		for to, e in v_frm.edges.items():
			if v_frm.label <= g.vertices[to].label:		# Chỉ lấy các cạnh có nhãn đỉnh frm <= nhãn đỉnh to 
				result.append(e)
			# if v_frm.label <= g.vertices[to].label and e.frm < e.to:		# Tránh lấy cạnh lặp lại
				# result.append(e)
		return result

	def get_backward_edge(self, g, e1, e2, history):
		"""Lấy cạnh backward"""
		if e1 == e2:	# Cùng 1 cạnh
			return None
		for to, e in g.vertices[e2.to].edges.items():	# duyệt các cạnh xuất phát từ đinh cuối 2
			if e.to != e1.frm or history.has_edge(e.id) :	# cạnh đã được duyệt hoặc không nối với đỉnh đầu cạnh 1 thì bỏ qua
				continue						
			if e1.label < e.label or (e1.label == e.label and g.vertices[e1.to].label <= g.vertices[e2.to].label):	# chỉ lấy 1 cạnh đúng thứ tự DFS
				return e			
		return None

	def get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
		result = []
		# Duyệt các cạnh nối từ đỉnh bên phải nhất
		for to, e in g.vertices[rm_edge.to].edges.items():
			# Chỉ lấy các cạnh chưa được duyệt hoặc có nhãn của đỉnh đến lớn hơn nhãn đỉnh v0 cho đúng thứ tự DFS
			if min_vlb <= g.vertices[e.to].label and (
					not history.has_vertex(e.to)):
				result.append(e)
		return result

	def get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history):
		result = []
		to_vlb = g.vertices[rm_edge.to].label
		# Duyệt các cạnh trên đỉnh trong đường dẫn bên phải không phải đỉnh bên phải nhất
		for to, e in g.vertices[rm_edge.frm].edges.items():
			new_to_vlb = g.vertices[to].label
			# Nếu đã duyệt hoặc là đỉnh backward thì bỏ qua
			if (rm_edge.to == e.to or
					min_vlb > new_to_vlb or
					history.has_vertex(e.to)):
				continue
			# Chỉ lấy cạnh đúng thứ tự DFS
			if rm_edge.label < e.label or (rm_edge.label == e.label and
									   to_vlb <= new_to_vlb):
				result.append(e)				
		return result


	def get_support(self, projected):
		""" Đếm số đồ thị chứa đồ thị con - Độ hỗ trợ của đồ thị con """
		return len(set([pdfs.gid for pdfs in projected]))

	def report(self, projected):
		self.frequent_subgraphs.append(copy.copy(self.DFScode))
		
		g = self.DFScode.to_graph(gid=next(self.counter))
		display_str = g.display()
		print('\nSupport: {}'.format(self.support))

		# Add some report info to pandas dataframe "self._report_df".
		self.report_df = self.report_df.append(
			pd.DataFrame(
				{
					'support': [self.support],
					'description': [display_str],
					'num_vert': self.DFScode.get_num_vertices()
				},
				index=[int(repr(self.counter)[6:-1])]
			)
		)
		
		g.plot()
		print('\n-----------------\n')

	def is_min(self):
		"""Kiểm tra DFSCode là nhỏ nhất chưa."""		
		if len(self.DFScode) == 1:
			return True
		g = self.DFScode.to_graph(gid=VACANT_GRAPH_ID)	# Chuyển mã DFSCode thành dạng đồ thị
		dfs_code_min = DFSCode()
		root = collections.defaultdict(Projected)
		for vid, v in g.vertices.items():
			# Duyệt từng đỉnh
			edges = self.get_forward_root_edges(g, v)	#Lấy cách cạnh 
			for e in edges:
				root[(v.label, e.label, g.vertices[e.to].label)].append(
					PDFS(g.id, e, None))
		min_vevlb = min(root.keys())		# Nhãn của cạnh nhỏ nhất 
		dfs_code_min.append(DFSEdge(0, 1, min_vevlb))	# Xuất phát từ cạnh nhỏ nhất

		def project_is_min(projected):
			dfs_code_min.build_rmpath()		# Lấy đường dãn bên phải
			rmpath = dfs_code_min.rmpath
			min_vlb = dfs_code_min[0].vevLabel[0]	# Nhãn đỉnh đầu
			maxtoc = dfs_code_min[rmpath[0]].to 	# Đỉnh bên phải nhất

			backward_root = collections.defaultdict(Projected)
			flag, newto, end = False, 0, 0			
			for i in range(len(rmpath) - 1, end, -1):
				if flag:
					break
				for p in projected:
					history = History(g, p)			# Lấy các cạnh đã duyệt qua
					# Tìm cạnh backward nhỏ nhất
					e = self.get_backward_edge(g,
												history.edges[rmpath[i]],
												history.edges[rmpath[0]],
												history)
					if e is not None:
						backward_root[e.label].append(PDFS(g.id, e, p))
						newto = dfs_code_min[rmpath[i]].frm
						flag = True
			if flag:	# tìm được cạnh backward nhỏ nhất
				backward_min_elb = min(backward_root.keys())	# Nhãn cạnh nhỏ nhất
				# Thêm cạnh backward vào đồ thị 
				dfs_code_min.append(DFSEdge(
					maxtoc, newto,
					(VACANT_VERTEX_LABEL,
					 backward_min_elb,
					 VACANT_VERTEX_LABEL)
				))
				idx = len(dfs_code_min) - 1
				# Nếu mã cạnh mới thêm khác mã trong đồ thị con tương ứng thì không phải DFSCode không phải là min
				if self.DFScode[idx] != dfs_code_min[idx]:
					return False
				# Ngược lại kiểm tra tiếp 
				return project_is_min(backward_root[backward_min_elb])

			forward_root = collections.defaultdict(Projected)
			flag, newfrm = False, 0
			for p in projected:
				history = History(g, p)
				# Lấy các cạnh forward từ đỉnh bên phải nhất
				edges = self.get_forward_pure_edges(g,
													 history.edges[rmpath[0]],
													 min_vlb,
													 history)
				if len(edges) > 0:
					flag = True
					newfrm = maxtoc
					for e in edges:
						forward_root[
							(e.label, g.vertices[e.to].label)
						].append(PDFS(g.id, e, p))
			for rmpath_i in rmpath:
				if flag:
					break
				for p in projected:
					history = History(g, p)
					# Lấy cách cạnh forward từ các đỉnh trên dường dẫn bên phải
					edges = self.get_forward_rmpath_edges(g,
														   history.edges[
															   rmpath_i],
														   min_vlb,
														   history)
					if len(edges) > 0:
						flag = True
						newfrm = dfs_code_min[rmpath_i].frm
						for e in edges:
							forward_root[
								(e.label, g.vertices[e.to].label)
							].append(PDFS(g.id, e, p))

			if not flag:	# Nếu không có đỉnh forward thì đúng là min
				return True
			# Nếu có đỉnh forward thì kiểm tra tiếp
			forward_min_evlb = min(forward_root.keys())
			dfs_code_min.append(DFSEdge(
				newfrm, maxtoc + 1,
				(VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1]))
			)
			idx = len(dfs_code_min) - 1
			if self.DFScode[idx] != dfs_code_min[idx]: # Nếu mã cạnh mới thêm khác mã trong đồ thị con tương ứng thì không phải DFSCode không phải là min
				return False
			# Kiểm tra tiếp
			return project_is_min(forward_root[forward_min_evlb])

		res = project_is_min(root[min_vevlb])
		return res
	
	@record_timestamp
	def run(self):
		""" Chạy giải thuật. """		
		self.readGraphs()
		self.get_frequent_1edge_subgraphs()
		root = collections.defaultdict(Projected)		
		for gid, g in self.graphs.items():
			for vid, v in g.vertices.items():
				edges = self.get_forward_root_edges(g, v)
				for e in edges:
					pdf = PDFS(gid, e, None)
					root[(v.label, e.label, g.vertices[e.to].label)].append(pdf)
				# 	pdf.display()
				# 	print('({},{},{})'.format(v.label, e.label, g.vertices[e.to].label))
				# print('-----------')
		# for (vlb1, elb, vlb2), pdfs in root.items():
		# 	print('root[({},{},{})]'.format(vlb1, elb, vlb2))
		# 	print(pdfs)

		self.ri = 0
		self.gc = 0
		for vevlb, projected in root.items():
			self.DFScode.append(DFSEdge(0, 1, vevlb))	# lấy từng đồ thị con 1 cạnh
			self.subgraph_mining(projected)				# tìm các đồ thị con phổ biến từ 1 cạnh trên
			self.DFScode.pop()							# Loại cạnh trên ra khỏi 

	def subgraph_mining(self, projected):
		# print('ri = ',self.ri)
		# self.ri += 1
		self.support = self.get_support(projected)	# Tính độ hỗ trợ của đồ thị con
		if self.support < self.minSup:				# nếu nhỏ hơn minSup thì bỏ qua
			return
		if not self.is_min():						# nếu mã DFSCode không nhỏ nhất thì bỏ qua
			return
		self.report(projected)						# Ghi nhận đồ thị con phổ biến
		# print(' gc = ',self.gc)
		# self.gc += 1
		# print(self.DFScode)
		# print(projected)

		self.DFScode.build_rmpath()				# tính đường dẫn bên phải
		rmpath = self.DFScode.rmpath
		maxtoc = self.DFScode[rmpath[0]].to 	# đỉnh bên phải nhất
		min_vlb = self.DFScode[0].vevLabel[0]		# nhãn đỉnh đầu của cạnh nhỏ nhất

		forward_root = collections.defaultdict(Projected)
		backward_root = collections.defaultdict(Projected)
		for p in projected:				# Duyệt từng hình chiếu của đồ thị con (cạnh) p
			g = self.graphs[p.gid]		# Lấy đồ thị g chứa đồ thị con p
			history = History(g, p)		# Lấy các cạnh đã duyệt qua trong g
			# duyệt các cạnh backward trước
			for rmpath_i in rmpath[::-1]:	# duyệt các cạnh backward xuất phát đỉnh bên phải nhất đến các đỉnh đã duyệt theo thứ tự từ nhỏ đến lớn
				e = self.get_backward_edge(g,
											history.edges[rmpath_i],
											history.edges[rmpath[0]],
											history)
				if e is not None:		
					backward_root[
						(self.DFScode[rmpath_i].frm, e.label)
					].append(PDFS(g.id, e, p))
			# Lấy các cạnh forward từ đỉnh bên phải nhất			
			edges = self.get_forward_pure_edges(g,
												 history.edges[rmpath[0]],
												 min_vlb,
												 history)
			for e in edges:
				forward_root[
					(maxtoc, e.label, g.vertices[e.to].label)
				].append(PDFS(g.id, e, p))
			# Lấy các cạnh forward từ đường dẫn bên phải
			for rmpath_i in rmpath:
				edges = self.get_forward_rmpath_edges(g,
													   history.edges[rmpath_i],
													   min_vlb,
													   history)
				for e in edges:
					forward_root[
						(self.DFScode[rmpath_i].frm,
						 e.label, g.vertices[e.to].label)
					].append(PDFS(g.id, e, p))

		# Duyệt từng đồ thị con được thêm 1 cạnh backward
		for to, elb in backward_root:
			self.DFScode.append(DFSEdge(
				maxtoc, to,
				(VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
			)
			self.subgraph_mining(backward_root[(to, elb)])
			self.DFScode.pop()
		# Duyệt từng đồ thị con được thêm 1 cạnh forward
		for frm, elb, vlb2 in forward_root:
			dfse = DFSEdge(
				frm, maxtoc + 1,
				(VACANT_VERTEX_LABEL, elb, vlb2))			
			self.DFScode.append(dfse)
			# dfse.display()
			self.subgraph_mining(forward_root[(frm, elb, vlb2)])
			self.DFScode.pop()

		return self



