export Graph, graph
export predict
export forward
export backward
export update
export zerograds
export clear



"""
    mutable struct Graph
# Field
`backward`: store backward operations
"""
mutable struct Graph
    # 存储反向传播操作
    backward::Vector
    function Graph()
        new(Vector(undef,0))
    end
end


import Base.size
import Base.length
import Base.getindex
import Base.deepcopy

Base.size(g::Graph)           = size(g.backward)
Base.size(g::Graph, dim::Int) = size(g.backward, dim)
Base.length(g::Graph)         = length(g.backward)
Base.getindex(g::Graph, k...) =  g.backward[k...]


# -- 全局图变量，存储所有反向运算的中间
# -- 变量，但是在参数更新后需要将其置空
global graph = Graph()


function clear()
    graph.backward = [];
end


function backward()
    for i = length(graph):-1:1
        graph[i]()
    end
    clear()
end
