#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int INF = 1e9 + 7;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll n, m;
    cin >> n >> m;
    vector<ll> a(n + 2, 0);
    for (ll i = 1; i <= n; i++) {
        cin >> a[i];
    }

    vector<ll> diff(n + 2, 0);
    while (m--) {
        ll l, r, k;
        cin >> l >> r >> k;
        diff[r] += k;
        diff[l - 1] -= k;
    }

    for (ll i = n; i >= 1; i--) {
        diff[i] += diff[i + 1];
        a[i] += diff[i];
    }

    for (ll i = 1; i <= n; i++) {
        cout << a[i] << " ";
    }

    return 0;
}