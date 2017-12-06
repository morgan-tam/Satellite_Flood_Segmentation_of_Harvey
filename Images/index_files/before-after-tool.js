// This tool implements the before and after component from baseplate components
// It populates images from URL parameters of "before" and "after"
// Images must be located at /17200/experiments/usatoday/responsive/before-after-tool

var get_query_params = function() {
    var query_string = {},
        query = window.location.search.substring(1),
        vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        var pair = vars[i].split("=");
        if (typeof query_string[pair[0]] === "undefined") {
            query_string[pair[0]] = decodeURIComponent(pair[1]);
        }
        else if (typeof query_string[pair[0]] === "string") {
            query_string[pair[0]] = [query_string[pair[0]], decodeURIComponent(pair[1])];
        }
        else {
            query_string[pair[0]].push(decodeURIComponent(pair[1]));
        }
    }
    return query_string;
};

var params = get_query_params();
var base_url = 'http://www.gannett-cdn.com/experiments/usatoday/responsive/before-after-tool/images/';

if (params.before && params.after) {
    var before_after = new BeforeAfter('.before-and-after', [
        base_url + params.before,
        base_url + params.after
    ]);
}