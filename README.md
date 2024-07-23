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

```cpp

```
```cpp

```
```cpp

```

