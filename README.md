MO : 

```cpp
void remove(idx);  // TODO: remove value at idx from data structure 

void add(idx);     // TODO: add value at idx from data structure 

int get_answer();  // TODO: extract the current answer of the data structure 

  

int block_size; 

  

struct Query { 

    int l, r, idx; 

    bool operator<(Query other) const 

    { 

        return make_pair(l / block_size, r) < 

               make_pair(other.l / block_size, other.r); 

    } 

}; 

  

vector<int> mo_s_algorithm(vector<Query> queries) { 

    vector<int> answers(queries.size()); 

    sort(queries.begin(), queries.end()); 

  

    // TODO: initialize data structure 

  

    int cur_l = 0; 

    int cur_r = -1; 

    // invariant: data structure will always reflect the range [cur_l, cur_r] 

    for (Query q : queries) { 

        while (cur_l > q.l) { 

            cur_l--; 

            add(cur_l); 

        } 

        while (cur_r < q.r) { 

            cur_r++; 

            add(cur_r); 

        } 

        while (cur_l < q.l) { 

            remove(cur_l); 

            cur_l++; 

        } 

        while (cur_r > q.r) { 

            remove(cur_r); 

            cur_r--; 

        } 

        answers[q.idx] = get_answer(); 

    } 

    return answers; 

} 
```
 

Dijsksta: 

```cpp

vector<vector<pair<int, int>>> adj; 

  

void dijkstra(int s, vector<int> & d, vector<int> & p) { 

    int n = adj.size(); 

    d.assign(n, INF); 

    p.assign(n, -1); 

  

    d[s] = 0; 

    set<pair<int, int>> q; 

    q.insert({0, s}); 

    while (!q.empty()) { 

        int v = q.begin()->second; 

        q.erase(q.begin()); 

  

        for (auto edge : adj[v]) { 

            int to = edge.first; 

            int len = edge.second; 

  

            if (d[v] + len < d[to]) { 

                q.erase({d[to], to}); 

                d[to] = d[v] + len; 

                p[to] = v; 

                q.insert({d[to], to}); 

            } 

        } 

    } 

}
```
