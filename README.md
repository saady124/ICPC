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

bool cmp(pair<int, int> p, pair<int, int> q) {
    if (p.first / BLOCK_SIZE != q.first / BLOCK_SIZE)
        return p < q;
    return (p.first / BLOCK_SIZE & 1) ? (p.second < q.second) : (p.second > q.second);
}
```
 

Dijsksta using SET: 

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

Dijsksta using PQ: 
``` cpp
void dijkstra(int s, vector<int> & d, vector<int> & p) {
    int n = adj.size();
    d.assign(n, INF);
    p.assign(n, -1);

    d[s] = 0;
    using pii = pair<int, int>;
    priority_queue<pii, vector<pii>, greater<pii>> q;
    q.push({0, s});
    while (!q.empty()) {
        int v = q.top().second;
        int d_v = q.top().first;
        q.pop();
        if (d_v != d[v])
            continue;

        for (auto edge : adj[v]) {
            int to = edge.first;
            int len = edge.second;

            if (d[v] + len < d[to]) {
                d[to] = d[v] + len;
                p[to] = v;
                q.push({d[to], to});
            }
        }
    }
}
```

DSU

```cpp
void make_set(int v) {
    parent[v] = v;
    size[v] = 1;
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (size[a] < size[b])
            swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
}
int find_set(int v) {
    if (v == parent[v])
        return v;
    return parent[v] = find_set(parent[v]);
}
```
SUFFIX Array
```cpp
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;

#define elsady ios_base::sync_with_stdio(false);cin.tie(nullptr);cout.tie(nullptr);
#define test int t;cin>>t;while(t--)
#define fs first
#define sc second
#define ll long long
#define endl '\n'
#define en end()
#define be begin()
#define no cout<<"NO"<<endl
#define yes cout<<"YES"<<endl
//#define int ll
#define gr greater<int>
#define pii pair<int,int>
#define tup tuple<int,int,int>
template <typename T> // T -> (can be integer, float or pair of int etc.)
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

const ll MOD1=998244353,OO=1e18,MOD=1e9+7;
const double PI=acos(-1);
const int N=5*(1e5+5),T=1e6+1;
vector<vector<int>>adj;
bool vis[N];
int dep[N],dis[N];
int dx[]= {1,-1,0,0,1,1,-1,-1};
int dy[]= {0,0,1,-1,1,-1,1,-1};
int dz[]= {0,0,0,0,1,-1};
int n,m;

// bitmasks
//Number of leading zeroes: __builtin_clz(x)
//Number of trailing zeroes : __builtin_ctz(x)
//Number of 1-bits: __builtin_popcount(x);
//last one : __lg()

ll gcd(ll x,ll y) {
    return y?gcd(y,x%y):x;
}

ll lcm(ll x,ll y) {
    return x/gcd(x,y)*y;
}

ll prom(ll x,ll y,ll mod) {
    return ((x%mod)*(y%mod))%mod;
}

ll inv(ll a,ll b) {
    return 1<a?b-inv(b%a,a)*b/a:1;
}

ll summ(ll a,ll b,ll mod){
    return ((a%mod)+(b%mod))%mod;
}

int fastpow(int x,int y,int mod)
{
   if(y==0) return 1;
   int c=fastpow(prom(x,x,mod),y/2,mod);
   if(y%2)c=prom(c,x,mod);
   return c;
}

bool prime(ll x)
{
    if(x<2)
        return 0;
    for(ll I=2; I*I<=x; I++)
        if(x%I==0)
            return 0;
    return 1;
}

bool allow(int I,int j) {
    return (I>=0&&j>=0&&I<n&&j<m);
}
void counting_sort(vector<int>&p,vector<int>&c)
{
      int n=p.size();
      vector<int>freq(n),pos(n);
      for(int i:c)++freq[i];
      pos[0]=0;
      for(int i=1;i<n;i++)pos[i]=pos[i-1]+freq[i-1];
      vector<int>pp(n);
      for(int i:p)pp[pos[c[i]]++]=i;
      p=pp;
}
void solve()
{
    string s;
    cin>>s;
    s+='$';
    int n=s.size();
    pair<char,int>a[n];
    vector<int>c(n);
    for(int i=0;i<n;i++)a[i]={s[i],i};
    sort(a,a+n);
    vector<int>p(n);
    for(int i=0;i<n;i++)p[i]=a[i].sc;
    c[p[0]]=0;
    for(int i=1;i<n;i++)
     c[p[i]]=c[p[i-1]]+(a[i].fs!=a[i-1].fs);
    int k=0;
    while((1<<k)<n)
    {
        for(int i=0;i<n;i++)p[i]=(p[i]-(1<<k)+n)%n;
        counting_sort(p,c);
        vector<int>cc(n);
        cc[p[0]]=0;
        for(int i=1;i<n;i++)
        {
            pii now={c[p[i]],c[(p[i]+(1<<k))%n]};
            pii prev={c[p[i-1]],c[(p[i-1]+(1<<k))%n]};
            cc[p[i]]=cc[p[i-1]]+(now!=prev);
        }
        k++;
        c=cc;
    }
    vector<int>lcp(n-1);
    int kk=0;
    for(int i=0;i<n-1;i++)
    {
        while(s[i+kk]==s[p[c[i]-1]+kk])kk++;
        lcp[c[i]-1]=kk--;
        kk=max(kk,0);
    }
    ll ans=0;
    for(int i=0;i<n-1;i++)ans+=n-p[i+1]-1-lcp[i];
    cout<<ans<<endl;
}
signed main()
{
    elsady
   // test
       solve();
}
```
Sparse Table

```cpp
struct sparse
{
    vector<int>lg;
    vector<int>sap[20];
    void init(int n)
    {
         for(int i=0;i<20;i++)sap[i].resize(n+1);
         lg.resize(n+1);
         lg[1]=0;
         for(int i=2;i<n;i++)lg[i]=lg[i/2]+1;
    }
    int mrg(int a,int b)
    {
        
    }
    void build(vector<int>a)
    {
       //  copy(a.begin(),a.end(),sap[0]);
         for(int i=0;i<a.size();i++)sap[0][i]=a[i];
         for(int i=1;i<20;i++)
             for(int j=0;j+(1<<(i-1))<a.size();j++)
               sap[i][j]=mrg(sap[i-1][j],sap[i-1][j+(1<<(i-1))]);
    }
    int query1(int l,int r)
    {
         int i=lg[r-l+1];
         return mrg(sap[i][l],sap[i][r-(1<<i)+1]);
    }
    int querylg(int l,int r)
    {
        int ans=0;
        for(int i=20;i>=0;i--)
        {
              if((1<<i)<=r-l+1)
              {
                  ans=mrg(ans,sap[i][l]);
                  l+=1<<i;
              }
        }
        return ans;
    }
};
```

LCA

```cpp
struct LCA {
    vector<int> height, euler, first, segtree;
    vector<bool> visited;
    int n;

    LCA(vector<vector<int>> &adj, int root = 0) {
        n = adj.size();
        height.resize(n);
        first.resize(n);
        euler.reserve(n * 2);
        visited.assign(n, false);
        dfs(adj, root);
        int m = euler.size();
        segtree.resize(m * 4);
        build(1, 0, m - 1);
    }

    void dfs(vector<vector<int>> &adj, int node, int h = 0) {
        visited[node] = true;
        height[node] = h;
        first[node] = euler.size();
        euler.push_back(node);
        for (auto to : adj[node]) {
            if (!visited[to]) {
                dfs(adj, to, h + 1);
                euler.push_back(node);
            }
        }
    }

    void build(int node, int b, int e) {
        if (b == e) {
            segtree[node] = euler[b];
        } else {
            int mid = (b + e) / 2;
            build(node << 1, b, mid);
            build(node << 1 | 1, mid + 1, e);
            int l = segtree[node << 1], r = segtree[node << 1 | 1];
            segtree[node] = (height[l] < height[r]) ? l : r;
        }
    }

    int query(int node, int b, int e, int L, int R) {
        if (b > R || e < L)
            return -1;
        if (b >= L && e <= R)
            return segtree[node];
        int mid = (b + e) >> 1;

        int left = query(node << 1, b, mid, L, R);
        int right = query(node << 1 | 1, mid + 1, e, L, R);
        if (left == -1) return right;
        if (right == -1) return left;
        return height[left] < height[right] ? left : right;
    }

    int lca(int u, int v) {
        int left = first[u], right = first[v];
        if (left > right)
            swap(left, right);
        return query(1, 0, euler.size() - 1, left, right);
    }
};
```
Hashing

```cpp
const int N = 1000001;
const int base1 = 31, mod1 = 1000000007, base2 = 37, mod2 = 2000000011;
int pw1[N], inv1[N], pw2[N], inv2[N];
pair<int, int> hashing[N], hashing_rev[N];
 
int add(int a, int b, int mod)
{
    int ans = a + b;
    if(ans >= mod)
        ans -= mod;
    if(ans < 0)
        ans += mod;
    return ans;
}
 
int mul(int a, int b, int mod)
{
    int ans = (a * 1ll * b) % mod;
    if(ans < 0)
        ans += mod;
    return ans;
}
 
int power(int a, int b, int mod) {
    if (b <= 0) return 1;
    int ret = power(mul(a, a, mod), b / 2, mod);
    if (b % 2) ret = mul(ret, a, mod);
    return ret;
}
 
void PreCalc() // Don't forget to call it before all the test cases
{
    pw1[0] = inv1[0] = pw2[0] = inv2[0] = 1;
    int pw_inv1 = power(base1, mod1 - 2, mod1);
    int pw_inv2 = power(base2, mod2 - 2, mod2);
 
    for (int i = 1; i < N; ++i) {
        pw1[i] = mul(pw1[i - 1], base1, mod1);
        inv1[i] = mul(inv1[i - 1], pw_inv1, mod1);
        pw2[i] = mul(pw2[i - 1], base2, mod2);
        inv2[i] = mul(inv2[i - 1], pw_inv2, mod2);
    }
}
 
void BuildHash(string s) // Pass to it the string before the queries, Must be passed be value
{
    int n = s.length();
    for (int i = 0; i < n; ++i) {
        hashing[i].first = add(((i == 0) ? 0 : hashing[i - 1].first), mul(pw1[i], s[i] - 'a' + 1, mod1), mod1);
        hashing[i].second = add(((i == 0) ? 0 : hashing[i - 1].second), mul(pw2[i], s[i] - 'a' + 1, mod2), mod2);
    }
 
    reverse(all(s));
    for (int i = 0; i < n; ++i) {
        hashing_rev[i].first = add(((i == 0) ? 0 : hashing_rev[i - 1].first), mul(pw1[i], s[i] - 'a' + 1, mod1), mod1);
        hashing_rev[i].second = add(((i == 0) ? 0 : hashing_rev[i - 1].second), mul(pw2[i], s[i] - 'a' + 1, mod2), mod2);
    }
}
 
pair<int, int> GetHash(int l, int r)
{
    pair<int, int> ans;
    ans.first = add(hashing[r].first, ((l==0)?0:-hashing[l - 1].first), mod1);
    ans.second = add(hashing[r].second, ((l==0)?0:-hashing[l - 1].second), mod2);
    ans.first = mul(ans.first, inv1[l], mod1);
    ans.second = mul(ans.second, inv2[l], mod2);
    return ans;
}
 
pair<int, int> GetHash_rev(int l, int r)
{
    pair<int, int> ans;
    ans.first = add(hashing_rev[r].first, ((l==0)?0:-hashing_rev[l - 1].first), mod1);
    ans.second = add(hashing_rev[r].second, ((l==0)?0:-hashing_rev[l - 1].second), mod2);
    ans.first = mul(ans.first, inv1[l], mod1);
    ans.second = mul(ans.second, inv2[l], mod2);
    return ans;
}
 
bool is_pal(int l, int r, int n) // use this function pass to it (l, r) 0 indexed, pass the length of the string
{
    return GetHash(l, r) == GetHash_rev(n - r - 1, n - l - 1);
}
```

Fast ncr

```cpp
void pre() {
    fact[0] = 1;
    for (ll i = 1; i < N; i++)
        fact[i] = fact[i - 1] * i, fact[i] %= mod;
    modinv[N - 1] = fastpow(fact[N - 1], mod - 2);
    for (ll i = N - 2; i >= 0; i--)
        modinv[i] = (i + 1) * modinv[i + 1] % mod;
 
}
 
ll ncr(ll n, ll r) {
    if (r > n || r < 0)
        return 0;
    return fact[n] * modinv[r] % mod * modinv[n - r] % mod;
}
```

ncr
```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define IO ios_base::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
#define all(x) x.begin(), (x).end()
#define el '\n'
#define sz(v) (int)(v).size()
#define yes cout << "YES" << el
#define no cout << "NO" << el
#define N (int)3000 + 4
ll mod = 1e9 + 7;
ll power(ll x, ll n)
{
    ll result = 1;
    while (n > 0)
    {
        if (n & 1 == 1) // y is odd
        {
            result = (result * x) % mod;
        }
        x = (x * x) % mod;
        n = n >> 1; // y=y/2;
    }
    return result;
}
ll fact[N];
ll inv(ll a)
{
    return power(a, mod - 2);
}

ll ncr(ll n, ll r)
{
    if (r > n)
        return 0;
    return (fact[n] * inv(fact[r] * fact[n - r] % mod)) % mod;
}
void solve()
{
    ll n, k, ans = 0;
    cin >> n >> k;
    fact[0] = 1;
    for (ll i = 1; i <= n + 50; i++)
        fact[i] = (i * fact[i - 1]) % mod;
    for (ll i = 0; i < k - 1; i++)
    {
        // cout << i << " " << ncr(k, i) << " " << (k - i) << ' ' << power(n - 1, k - 1 - i) << " ";
        if (i % 2)
            ans -= (((ncr(k, k - i) * (k - i)) % mod) * power(k - 1 - i, n - 1)) % mod;
        else
            ans += (((ncr(k, k - i) * (k - i)) % mod) * power(k - 1 - i, n - 1)) % mod;
        // cout << ans << el;
        while (ans < 0)
        {
            ans += mod;
        }
        ans %= mod;
    }
    while (ans < 0)
    {
        ans += mod;
    }

    ans %= mod;
    int a;
    for (ll i = 0; i < n - 1; i++)
        cin >> a;
    cout << ans << el;
}
int main()
{
    IO solve();

    return 0;
}
/*
3*2*2*2 = 24
2*1*1*1 = 2

*/
/*
rgrr
grgg
*2


rgb
3c2
*/
```

matrix power

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define IO ios_base::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
#define all(x) x.begin(), (x).end()
#define el '\n'
#define sz(v) (ll)(v).size()
#define yes cout << "YES" << el
#define no cout << "NO" << el
#define MOD 1000000007
#define N 201

void multiply(double long a[N][N], double long b[N][N])
{
    double long mul[N][N];
    for (ll i = 0; i < N; i++)
    {
        for (ll j = 0; j < N; j++)
        {
            mul[i][j] = 0;
            for (ll k = 0; k < N; k++)
                mul[i][j] = (mul[i][j] + (a[i][k] * b[k][j]));
        }
    }
    for (ll i = 0; i < N; i++)
        for (ll j = 0; j < N; j++)
            a[i][j] = mul[i][j];
}

void power(double long F[N][N], ll n)
{
    double long M[N][N];
    memset(M, 0, sizeof M);
    for (int i = 0; i < N; i++)
    {
        M[i][i] = 1;
    }
    // M[0][0] = 1;
    while (n > 0)
    {
        if (n & 1)
            multiply(M, F);
        multiply(F, F);
        n >>= 1;
    }

    for (ll i = 0; i < N; i++)
        for (ll j = 0; j < N; j++)
            F[i][j] = M[i][j];
}

void solve()
{
    ll n, m;
    cin >> n >> m;
    double long F[N][N];
    memset(F, 0, sizeof F);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; ++j)
            cin >> F[i][j];
    }
    power(F, m);
    /*for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; ++j)
            cout << F[i][j] << " ";
        cout << el;
    }
    cout << el;
    */
    cout << fixed << setprecision(6);

    for (int i = 0; i < n; i++)
        cout << F[0][i] << el;
}

int main()
{
    IO
    solve();
    return 0;
}
```

sieve

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define IO ios_base::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
#define all(x) x.begin(), (x).end()
#define el '\n'
#define sz(v) (int)(v).size()
#define yes cout << "YES" << el
#define no cout << "NO" << el
#define N (int)1000 + 4
vector<int> primes, freq(1e6, 0);
void SieveOfEratosthenes(int n)
{
    // Create a boolean array "prime[0..n]" and initialize
    // all entries it as true. A value in prime[i] will
    // finally be false if i is Not a prime, else true.
    bool prime[n + 1];
    memset(prime, true, sizeof(prime));

    for (int p = 2; p * p <= n; p++)
    {
        // If prime[p] is not changed, then it is a prime
        if (prime[p] == true)
        {
            // Update all multiples of p greater than or
            // equal to the square of it numbers which are
            // multiple of p and are less than p^2 are
            // already been marked.
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }

    // Print all prime numbers
    for (int p = 2; p <= n; p++)
        if (prime[p])
            primes.push_back(p);
}
void solve()
{
    ll n, m, a;
    cin >> n >> m;

    for (int i = 0; i < n; i++)
    {
        cin >> a;
        for (int j = 0; j < primes.size(); j++)
        {
            while (a && a % primes[j] == 0)
            {
                // cout << primes[j] << el;
                freq[primes[j]]++;
                a /= primes[j];
            }
        }

        if (a > 1)
            freq[a]++;
        // cout << freq[3] << " " << el;
    }
    for (int i = 2; i < 1e6; i++)
    {
        if (freq[i] % m)
        {
            // cout << i << " " << freq[i] << el;
            no;
            return;
        }
    }
    yes;
}
int main()
{
    IO;
    SieveOfEratosthenes(N);
    solve();

    return 0;
}
```

LIS

```cpp
int lis(vector<int> const &a)
{
    int n = a.size();
    const int INF = 1e9;
    vector<int> d(n + 1, INF);
    d[0] = -INF;
 
    for (int i = 0; i < n; i++)
    {
        int l = upper_bound(d.begin(), d.end(), a[i]) - d.begin();
        if (d[l - 1] < a[i] && a[i] < d[l])
            d[l] = a[i];
    }
 
    int ans = 0;
    for (int l = 0; l <= n; l++)
    {
        if (d[l] < INF)
            ans = l;
    }
    return ans;
}
```
standard segment tree
 
```cpp
struct segtree
{
    int size=1;
    vector<pii>mins,maxs;
    void init(int n)
    {
        while(size<n)size*=2;
        mins.assign(2*size,{MOD,MOD});
        maxs.assign(2*size,{0ll,0ll});
    }
    void build(vector<int>&a)
    {
        build(a,0,0,size);
    }
    void build(vector<int>&a,int x,int lx,int ry)
    {
        if(ry-lx==1)
        {
            if(lx<a.size())
                mins[x]={a[lx],lx},maxs[x]={a[lx],lx};
            return ;
        }
        int mid=(lx+ry)/2;
        build(a,2*x+1,lx,mid);
        build(a,2*x+2,mid,ry);
        mins[x]=min(mins[2*x+1],mins[2*x+2]);
        maxs[x]=max(maxs[2*x+1],maxs[2*x+2]);
    }
    void set(int i,int v)
    {
        set(i,v,0,0,size);
    }
    void set(int i,int v,int x,int lx,int ry)
    {
        if(ry-lx==1)
        {
            mins[x].fs=v,maxs[x].fs=v;
            return;
        }
        int mid=(lx+ry)/2;
        if(i<mid)
            set(i,v,2*x+1,lx,mid);
        else
            set(i,v,2*x+2,mid,ry);
        mins[x]=min(mins[2*x+1],mins[2*x+2]);
        maxs[x]=max(maxs[2*x+1],maxs[2*x+2]);
    }
    int minimum(int l,int r)
    {
        return minimum(l,r,0,0,size);
    }
    int minimum(int l,int r,int x,int lx,int ry)
    {
        if(lx>=r||ry<=l)return MOD;
        if(lx>=l&&ry<=r)return mins[x].fs;
        int mid=(lx+ry)/2;
        return min(minimum(l,r,2*x+1,lx,mid),minimum(l,r,2*x+2,mid,ry));
    }
    int maximum(int l,int r)
    {
        return maximum(l,r,0,0,size);
    }
    int maximum(int l,int r,int x,int lx,int ry)
    {
        if(lx>=r||ry<=l)return -MOD;
        if(lx>=l&&ry<=r)return maxs[x].fs;
        int mid=(lx+ry)/2;
        return max(maximum(l,r,2*x+1,lx,mid),maximum(l,r,2*x+2,mid,ry));
    }
    int max_v(int v)
    {
        return max_v(v,0,0,size);
    }
    int max_v(int v,int x,int lx,int ry)
    {
        if(maxs[x].fs<=v)return MOD;
        if(ry-lx==1)return maxs[x].sc;
        int mid=(lx+ry)/2;
        if(maxs[2*x+1].fs>v)return max_v(v,2*x+1,lx,mid);
        return max_v(v,2*x+2,mid,ry);
    }
    int min_v(int v)
    {
        return min_v(v,0,0,size);
    }
    int min_v(int v,int x,int lx,int ry)
    {
        if(mins[x].fs>=v)return -1;
        if(ry-lx==1)return mins[x].sc;
        int mid=(lx+ry)/2;
        if(mins[2*x+2].fs<v)return min_v(v,2*x+2,mid,ry);
        return min_v(v,2*x+1,lx,mid);
    }
};
```
Topological sort

```cpp
int n; // number of vertices
vector<vector<int>> adj; // adjacency list of graph
vector<bool> visited;
vector<int> ans;
void dfs(int v) {
 visited[v] = true;
 for (int u : adj[v]) {
 if (!visited[u])
 dfs(u);
 }
 ans.push_back(v);
}
void topological_sort() {
 visited.assign(n, false);
 ans.clear();
 for (int i = 0; i < n; ++i) {
 if (!visited[i]) {dfs(i);
 }
 }
 reverse(ans.begin(), ans.end());
}
```

MST

```cpp
int parent[N],siz[N],sum[N];
void make_set(int v) {
 parent[v] = v;
 siz[v] = 1;
}void init() {
 for (int i = 1; i < N; i++) {
 make_set(i);
 }
 return;
}
int fSet(int v) {
 if (v == parent[v])
 return v;
 return parent[v] = fSet(parent[v]);
}
void uSets(int a, int b) {
 a = fSet(a);
 b = fSet(b);if (a != b) {
 if (siz[a] < siz[b])
 swap(a, b);
 parent[b] = a;
 siz[a] += siz[b];
 }
}
struct Edge {
 int u, v, weight;
 bool operator<(Edge const& other) {
 return weight < other.weight;
 }
};
vector<Edge> edges;
//inside main
init();
int cost = 0;
vector<Edge> result;
sort(edges.begin(), edges.end());
for (Edge e : edges) {
 if (fSet(e.u) != fSet(e.v)) {
 cost += e.weight;
 result.push_back(e);
 uSets(e.u, e.v);
 }
}
```

0-1 BFS
```cpp
1 BFS
when there are at most two types of edges.
Dijkstra is in $O(|E| \log |V|)$ but this in $O(|E|)$.
vector<int> d(n, INF);
d[s] = 0;
deque<int> q;
q.push_front(s);
while (!q.empty()) {
 int v = q.front();
 q.pop_front();
 for (auto edge : adj[v]) {
 int u = edge.first;
 int w = edge.second;
 if (d[v] + w < d[u]) {
 d[u] = d[v] + w;
 if (w == 1)
 q.push_back(u);
 else
 q.push_front(u);
 }
 }
}
```

```cpp
// C++ program to find maximum XOR in
// a stream of integers
#include <bits/stdc++.h>
using namespace std;

struct TrieNode
{
    TrieNode *children[2];
    int cnt = 0;
    int isLeaf;
};

// This checks if the ith position in
// binary of N is a 1 or a 0
bool check(int N, int i)
{
    return (bool)(N & (1 << i));
}

// Create a new Trie node
TrieNode *newNode()
{
    TrieNode *temp = new TrieNode;
    temp->isLeaf = 0;
    temp->cnt = 0;
    temp->children[0] = NULL;
    temp->children[1] = NULL;
    return temp;
}

// Inserts x into the Trie
void query1(TrieNode *root, int x, int p)
{
    TrieNode *Crawler = root;

    // padding upto 32 bits
    for (int i = 31; i >= 0; i--)
    {
        int f = check(x, i);
        if (!Crawler->children[f])
            Crawler->children[f] = newNode();
        // Crawler->cnt++;
        Crawler->children[f]->cnt += p;
        Crawler = Crawler->children[f];
    }
    Crawler->isLeaf += p;
}

// Finds maximum XOR of x with stream of
// elements so far.
int query2(TrieNode *root, int x)
{
    TrieNode *Crawler = root;

    // Do XOR from root to a leaf path
    int ans = 0;
    for (int i = 31; i >= 0; i--)
    {
        // Find i-th bit in x
        int f = check(x, i);

        // Move to the child whose XOR with f
        // is 1.
        if ((Crawler->children[f ^ 1]) && (Crawler->children[f ^ 1]->cnt))
        {
            ans = ans + (1 << i); // update answer
            Crawler = Crawler->children[f ^ 1];
        }

        // If child with XOR 1 doesn't exist
        else
            Crawler = Crawler->children[f];
    }

    return ans;
}

// Driver code
int main()
{
    TrieNode *root = newNode();
    query1(root, 0, 1);
    int q;
    cin >> q;
    while (q--)
    {
        char c;
        int x;
        cin >> c >> x;
        if (c == '+')
        {
            query1(root, x, 1);
        }
        else if (c == '-')
        {
            query1(root, x, -1);
        }
        else
        {
            cout << query2(root, x) << '\n';
        }
    }
    return 0;
}
```


```cpp
#include <bits/stdc++.h>
using namespace std;

struct TrieNode {

    // pointer array for child nodes of each node
    TrieNode* childNode[26];

    // Used for indicating ending of string
    bool wordEnd;

    TrieNode()
    {
        // constructor
        // initialize the wordEnd variable with false
        // initialize every index of childNode array with
        // NULL
        wordEnd = false;
        for (int i = 0; i < 26; i++) {
            childNode[i] = NULL;
        }
    }
};

void insert_key(TrieNode* root, string& key)
{
    // Initialize the currentNode pointer
    // with the root node
    TrieNode* currentNode = root;

    // Iterate across the length of the string
    for (auto c : key) {

        // Check if the node exist for the current
        // character in the Trie.
        if (currentNode->childNode[c - 'a'] == NULL) {

            // If node for current character does not exist
            // then make a new node
            TrieNode* newNode = new TrieNode();

            // Keep the reference for the newly created
            // node.
            currentNode->childNode[c - 'a'] = newNode;
        }

        // Now, move the current node pointer to the newly
        // created node.
        currentNode = currentNode->childNode[c - 'a'];
    }

    // Increment the wordEndCount for the last currentNode
    // pointer this implies that there is a string ending at
    // currentNode.
    currentNode->wordEnd = 1;
}

bool search_key(TrieNode* root, string& key)
{
    // Initialize the currentNode pointer
    // with the root node
    TrieNode* currentNode = root;

    // Iterate across the length of the string
    for (auto c : key) {

        // Check if the node exist for the current
        // character in the Trie.
        if (currentNode->childNode[c - 'a'] == NULL) {

            // Given word does not exist in Trie
            return false;
        }

        // Move the currentNode pointer to the already
        // existing node for current character.
        currentNode = currentNode->childNode[c - 'a'];
    }

    return (currentNode->wordEnd == true);
}

// Driver code
int main()
{
    // Make a root node for the Trie
    TrieNode* root = new TrieNode();

    // Stores the strings that we want to insert in the
    // Trie
    vector<string> inputStrings
        = { "and", "ant", "do", "geek", "dad", "ball" };

    // number of insert operations in the Trie
    int n = inputStrings.size();

    for (int i = 0; i < n; i++) {
        insert_key(root, inputStrings[i]);
    }

    // Stores the strings that we want to search in the Trie
    vector<string> searchQueryStrings
        = { "do", "geek", "bat" };

    // number of search operations in the Trie
    int searchQueries = searchQueryStrings.size();

    for (int i = 0; i < searchQueries; i++) {
        cout << "Query String: " << searchQueryStrings[i]
             << "\n";
        if (search_key(root, searchQueryStrings[i])) {
            // the queryString is present in the Trie
            cout << "The query string is present in the "
                    "Trie\n";
        }
        else {
            // the queryString is not present in the Trie
            cout << "The query string is not present in "
                    "the Trie\n";
        }
    }

    return 0;
}
```
Trie :
```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define IO ios_base::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
#define all(x) x.begin(), (x).end()
#define el '\n'
#define sz(v) (int)(v).size()
#define yes cout << "Y" << el
#define no cout << "N" << el
#define N (int)1e6 + 4

struct TrieNode
{

    TrieNode *childNode[26];
    set<int> wordEnd, app;

    TrieNode()
    {
        for (int i = 0; i < 26; i++)
        {
            childNode[i] = NULL;
        }
    }
};

void insert_key(TrieNode *root, string &key, int idx)
{
    TrieNode *currentNode = root;
    for (auto c : key)
    {
        if (currentNode->childNode[c - 'a'] == NULL)
        {
            TrieNode *newNode = new TrieNode();
            currentNode->childNode[c - 'a'] = newNode;
        }
        currentNode = currentNode->childNode[c - 'a'];
        currentNode->app.emplace(idx);
    }

    currentNode->wordEnd.emplace(idx);
}
void delete_key(TrieNode *root, string &key, int idx)
{
    TrieNode *currentNode = root;
    // TrieNode *prev = root;
    for (auto c : key)
    {
        /*if (currentNode->childNode[c - 'a']->app.size() ==)
        {
            prev = currentNode;
            currentNode = currentNode->childNode[c - 'a'];
            prev->childNode[c - 'a']
        }
        else*/
        currentNode = currentNode->childNode[c - 'a'];
        currentNode->app.erase(idx);
    }

    currentNode->wordEnd.erase(idx);
}

bool search_key(TrieNode *root, string &key, int l, int r)
{
    // Initialize the currentNode pointer
    // with the root node
    TrieNode *currentNode = root;

    // Iterate across the length of the string
    for (auto c : key)
    {

        // Check if the node exist for the current
        // character in the Trie.
        if (currentNode->childNode[c - 'a'] == NULL)
        {

            // Given word does not exist in Trie
            return false;
        }

        // Move the currentNode pointer to the already
        // existing node for current character.
        currentNode = currentNode->childNode[c - 'a'];
    }
    auto itr = currentNode->app.lower_bound(l);
    if (itr == currentNode->app.end() || *itr > r)
        return false;
    return true;
    // return (currentNode->app.size() >= 1);
}
bool search_for_prefix(TrieNode *root, string &key, int l, int r)
{
    // Initialize the currentNode pointer
    // with the root node
    TrieNode *currentNode = root;
    // cout << key << " " << l << " " << r << el;
    //  Iterate across the length of the string
    for (auto c : key)
    {
        // cout << c << " " << l << " " << r << el;

        // Check if the node exist for the current
        // character in the Trie.
        if (currentNode->childNode[c - 'a'] == NULL)
        {

            // Given word does not exist in Trie
            return false;
        }
        currentNode = currentNode->childNode[c - 'a'];

        // Move the currentNode pointer to the already
        // existing node for current character.
        auto itr = currentNode->wordEnd.lower_bound(l);
        if (itr == currentNode->wordEnd.end() || *itr > r)
            continue;
        return true;
    }
    auto itr = currentNode->wordEnd.lower_bound(l);
    if (itr == currentNode->wordEnd.end() || *itr > r)
        return false;
    return true;
}
void solve()
{
    int n;
    cin >> n;
    string s, str[n + 1];
    TrieNode *root = new TrieNode();
    for (int i = 1; i <= n; i++)
    {
        cin >> str[i];
        insert_key(root, str[i], i);
    }
    int q, a, b, c;
    cin >> q;
    while (q--)
    {
        cin >> a;
        if (a == 1)
        {
            cin >> a >> s;
            delete_key(root, str[a], a);
            insert_key(root, s, a);
            str[a] = s;
        }
        else if (a == 2)
        {
            cin >> a >> b >> s;
            if (search_for_prefix(root, s, a, b))
                yes;
            else
                no;
        }
        else
        {
            cin >> a >> b >> s;
            if (search_key(root, s, a, b))
                yes;
            else
                no;
        }
    }
}
int main()
{
    // IO
    solve();

    return 0;
}
```
```cpp
#include <bits/stdc++.h>

using namespace std;

// This code is 
int count(int coins[], int n, int sum)
{
    // table[i] will be storing the number of solutions for
    // value i. We need sum+1 rows as the dp is
    // constructed in bottom up manner using the base case
    // (sum = 0)
    int dp[sum + 1];

    // Initialize all table values as 0
    memset(dp, 0, sizeof(dp));

    // Base case (If given value is 0)
    dp[0] = 1;

    // Pick all coins one by one and update the table[]
    // values after the index greater than or equal to the
    // value of the picked coin
    for (int i = 0; i < n; i++)
        for (int j = coins[i]; j <= sum; j++)
            dp[j] += dp[j - coins[i]];
    return dp[sum];
}

// Driver Code
int main()
{
    int coins[] = { 1, 2, 3 };
    int n = sizeof(coins) / sizeof(coins[0]);
    int sum = 5;
    cout << count(coins, n, sum);
    return 0;
}
```
Lazy :
```cpp
struct segtree
{
  int size=1;
  vector<int>sums,operations;
  void init(int n)
  {
     while(size<n)size*=2;
     sums.assign(2*size,0ll);
     operations.assign(2*size,OO);
  }
  void propygate(int l,int r,int x)
  {
     if(r-l==1||operations[x]==OO)
     {
        operations[x]=OO;
        return ;
     }
     int mid=(l+r)/2;
     operations[2*x+1]=operations[x];
     operations[2*x+2]=operations[x];
     sums[2*x+1]=operations[x]*(mid-l);
     sums[2*x+2]=operations[x]*(r-mid);
     operations[x]=OO;
  }
  void build(vector<int>&a)
  {
    build(a,0,0,size);
  }
  void build(vector<int>&a,int x,int lx,int ry)
  {
      if(ry-lx==1)
      {
        if(lx<a.size())
        sums[x]=a[lx];
        return ;
      }
      int mid=(lx+ry)/2;
      build(a,2*x+1,lx,mid);
      build(a,2*x+2,mid,ry);
      sums[x]=(sums[2*x+1]+sums[2*x+2]);
  }
  void modify(int l,int r,int v)
  {
    modify(l,r,v,0,0,size);
  }
  void modify(int l,int r,int v,int x,int lx,int ry)
  {
     propygate(lx,ry,x);
     if(ry<=l||lx>=r)return ;
     if(ry<=r&&lx>=l)
     {
         sums[x]=v*(ry-lx);
         operations[x]=v;
         return ;
     }
     int mid=(lx+ry)/2;
     modify(l,r,v,2*x+1,lx,mid);
     modify(l,r,v,2*x+2,mid,ry);
     sums[x]=(sums[2*x+1]+sums[2*x+2]);
  }
  int sum(int l,int r)
  {
    return sum(l,r,0,0,size);
  }
  int sum(int l,int r,int x,int lx,int ry)
  {
     propygate(lx,ry,x);
     if(lx>=r||ry<=l)return 0;
     if(lx>=l&&ry<=r)return sums[x];
     int mid=(lx+ry)/2;
     return (sum(l,r,2*x+1,lx,mid)+sum(l,r,2*x+2,mid,ry));
  }
};
```
XOR Segment:
```cpp
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;
#define elsady ios_base::sync_with_stdio(false);cin.tie(nullptr);cout.tie(nullptr);
#define test int t;cin>>t;while(t--)
#define fs first
#define sc second
#define ll long long
#define endl '\n'
#define be begin()
#define en end()
#define no cout<<"NO"<<endl
#define yes cout<<"YES"<<endl
#define int ll
#define gr greater<int>
#define pii pair<int,int>
template <typename T> // T -> (can be integer, float or pair of int etc.)
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
const ll MOD1=998244353,OO=1e18,MOD=1e9+7;
const double PI=acos(-1);
const int N=5*(1e5+5),T=1e6+1;
vector<vector<int>>adj;
bool vis[N];
int dep[N];
int dx[]= {1,-1,0,0,1,1,-1,-1};
int dy[]= {0,0,1,-1,1,-1,1,-1};
int dz[]= {0,0,0,0,1,-1};
int n,m;
int dis[N];
// bitmasks
//Number of leading zeroes: __builtin_clz(x)
//Number of trailing zeroes : __builtin_ctz(x)
//Number of 1-bits: __builtin_popcount(x);
//last one : __lg()
ll GCD(ll x,ll y) {
    return y?GCD(y,x%y):x;
}
ll LCM(ll x,ll y) {
    return x/GCD(x,y)*y;
}
ll prom(ll x,ll y,ll mod) {
    return ((x%mod)*(y%mod))%mod;
}
ll inv(ll a,ll b) {
    return 1<a?b-inv(b%a,a)*b/a:1;
}
int fastpow(int x,int y,int mod)
{
   if(y==0) return 1;
   int c=fastpow(prom(x,x,mod),y/2,mod);
   if(y%2)c=prom(c,x,mod);
   return c;
}
bool prime(ll x)
{
    if(x<2)
        return 0;
    for(ll I=2; I*I<=x; I++)
        if(x%I==0)
            return 0;
    return 1;
}
bool allow(int I,int j) {
    return (I>=0&&j>=0&&I<n&&j<m);
}
struct segtree
{
  int size=1;
  vector<int>sums[21];
  vector<int>operations;
  void init(int n)
  {
     while(size<n)size*=2;
     for(int i=0;i<=20;i++)sums[i].assign(2*size,0ll);
     operations.assign(2*size,0ll);
  }
  void propygate(int l,int r,int x)
  {
    if(r-l==1)
    {
        operations[x]=0;
        return ;
    }
    operations[2*x+1]^=operations[x];
    operations[2*x+2]^=operations[x];
    int mid=(l+r)/2;
    for(int i=0;i<=20;i++)
    {
        if(operations[x]&(1<<i))
        {
           sums[i][2*x+1]=mid-l-sums[i][2*x+1];
           sums[i][2*x+2]=r-mid-sums[i][2*x+2];
        } 
    }
    operations[x]=0;
  }
  void build(vector<int>&a)
  {
      build(a,0,0,size);
  }
  void build(vector<int>&a,int x,int lx,int ry)
  {
       if(ry-lx==1)
       {
          if(lx<a.size())
          for(int i=0;i<=20;i++)
           sums[i][x]=((a[lx]&(1<<i))!=0);
          return ;
       }
       int mid=(lx+ry)/2;
       build(a,2*x+1,lx,mid);
       build(a,2*x+2,mid,ry);
       for(int i=0;i<=20;i++)
        sums[i][x]=sums[i][2*x+1]+sums[i][2*x+2];
  }
  void modify(int l,int r,int v)
  {
    modify(l,r,v,0,0,size);
  }
  void modify(int l,int r,int v,int x,int lx,int ry)
  {
     propygate(lx,ry,x);
     if(ry<=l||lx>=r)return ;
     if(ry<=r&&lx>=l)
     {
         for(int i=0;i<=20;i++)
           if(v&(1<<i))
         sums[i][x]=ry-lx-sums[i][x];
         operations[x]=v;
         return ;
     }
     int mid=(lx+ry)/2;
     modify(l,r,v,2*x+1,lx,mid);
     modify(l,r,v,2*x+2,mid,ry);
     for(int i=0;i<=20;i++)
      sums[i][x]=sums[i][2*x+1]+sums[i][2*x+2];
  }
  int sum(int l,int r)
  {
    return sum(l,r,0,0,size);
  }
  int sum(int l,int r,int x,int lx,int ry)
  {
     propygate(lx,ry,x);
     if(lx>=r||ry<=l)return 0;
     if(lx>=l&&ry<=r)
     {
        int ans=0;
        for(int i=0;i<=20;i++)
          ans+=sums[i][x]*(1<<i);
        return ans;
     }
     int mid=(lx+ry)/2;
     return (sum(l,r,2*x+1,lx,mid)+sum(l,r,2*x+2,mid,ry));
  }
};
void solve()
{
    int n;
    cin>>n;
    vector<int>a(n);
    for(int &i:a)cin>>i;
    segtree st;
    st.init(n+1);
    st.build(a);
    int q;
    cin>>q;
    while (q--)
    {
        int t;
        cin>>t;
        if(t==2)
        {
            int l,r,v;
            cin>>l>>r>>v;
            st.modify(l-1,r,v);
        }
        else
        {
            int l,r;
            cin>>l>>r;
            cout<<st.sum(l-1,r)<<endl;    
        } 
    }
}
signed main()
{
    elsady
    //test
       solve();
}
```
