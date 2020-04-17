var ctx = document.getElementById('trend').getContext('2d');
var filterList = document.getElementById("filter");
var filterSel = filterList.options[filterList.selectedIndex].value;
var filterYearList = document.getElementById("filter-year");
var filterYearSel = filterYearList.options[filterYearList.selectedIndex].value;

var xAxisData = [];
var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

console.log(filterYearSel);
if (filterSel === "year") {
    for (var j = 11; j >= 0; j--) {
        xAxisData.push((filterYearSel - j).toString());
    }
}
else if (filterSel === "month") {
    for (j = 11; j >= 0; j--) {
        xAxisData = months;
    }
}


console.log(data[0]);
var dataset = [];
var colors = ['rgb(255, 99, 132)', 'rgb(255, 159, 64)', 'rgb(255, 159, 64)', 'rgb(255, 159, 64)', 'rgb(255, 159, 64)'];
for (var i = 0; i < 5; i++) {
    dataset.push(
        {
            label: data[i].word,
            backgroundColor: colors[i],
            borderColor: colors[i],
            data: [{x: 'b', y: 10}, {x: 'c', y: 19}, {x: 'd', y: 50}],
            fill: false,
            lineTension: 0
        }
    );
}

var myChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: xAxisData,
        datasets: dataset,
    },
    options: {
        responsive: true,
        tooltips: {
            mode: 'index',
            intersect: false
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            xAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Time'
                }
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'Frequency'
                },
            }]
        }
    }
});