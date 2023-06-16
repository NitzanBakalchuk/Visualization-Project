import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Specify the backend explicitly, e.g., 'Agg'
import plotly.express as px

import matplotlib.pyplot as plt



df15 = pd.read_csv("2015.csv")
df16 = pd.read_csv("2016.csv")
df17 = pd.read_csv("2017.csv")
df18 = pd.read_csv("2018.csv")
df19 = pd.read_csv("2019.csv")
df = pd.read_csv("df.csv")

# Define UI
def app():
################################## world map #########################################################
    st.title("World Happiness visualization")


    st.header("World Map -Happiness Score Per Country")
    # df = pd.read_csv("df.csv")

    # Create the choropleth map
    fig = px.choropleth(df, locations='Country', locationmode='country names',
                        color='Rank',
                        title='Rank of the countries',
                        labels={'Value': 'Score', 'Country': 'Country'},
                        hover_name='Country',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        hover_data={'Rank': True, 'Country': True, 'Score': True},
                        projection='natural earth')

    # Display the map
    st.plotly_chart(fig, use_container_width=True)
########################  Happiness ranking  ################################################################

    st.header("Select a country and find out why ")
    st.header("people are 'happy' there ")

    st.write('Explore the difference in Happiness ranking of countries between the selected years')
    years = ['2015', '2016', '2017', '2018', '2019']
    all_country = df15['Country'].unique().tolist()
    selected_country = st.selectbox("Select a country", [None] + all_country)
    selected_year_1 = st.selectbox("Select Year 1", [None] + years)
    selected_year_2 = st.selectbox("Select Year 2", [None] + years)
    df1 = df15
    df2 = df19

    if selected_year_1 == '2015':
        df1 = df15
    elif selected_year_1 == '2016':
        df1 = df16
    elif selected_year_1 == '2017':
        df1 = df17
    elif selected_year_1 == '2018':
        df1 = df18
    elif selected_year_1 == '2019':
        df1 = df19

    if selected_year_2 == '2015':
        df2 = df15
    elif selected_year_2 == '2016':
        df2 = df16
    elif selected_year_2 == '2017':
        df2 = df17
    elif selected_year_2 == '2018':
        df2 = df18
    elif selected_year_2 == '2019':
        df2 = df19

    dfall = pd.merge(df1[['Country', 'Rank']], df2[['Country', 'Rank']], on='Country')
    dfall.columns = ['Country', 'Rank Year 1', 'Rank Year 2']
    dfall['Rank Change'] = dfall['Rank Year 1'] - dfall['Rank Year 2']


    if selected_country is None:
        st.dataframe(
            dfall.style.apply(lambda x: ['background: red' if x['Rank Change'] < 0 else 'background: green' for _ in x],
                              axis=1))

    else:
        st.subheader("Rank Table")
        filtered_df = dfall[dfall['Country'] == selected_country]
        filtered_df = filtered_df[['Country', 'Rank Year 1', 'Rank Year 2', 'Rank Change']]

        st.dataframe(filtered_df)

#####################################################################################################
     st.header('3 most influential features through the years by contry')
    st.write('Here you can see the difference between the impact of the fetures through the years by the contries that rank highest and lowest ')

    features = ['Economy', 'Family', 'Health']
    ii = ['Switzerland', 'Iceland', 'Denmark', 'Norway', 'Canada', 'Finland', 'Togo', 'Burundi', 'Syria', 'Burkina Faso', 'Afghanistan']

    dfs = {'2015': df15[df15['Country'].isin(ii)][['Country'] + features],
           '2016': df16[df16['Country'].isin(ii)][['Country'] + features],
           '2017': df17[df17['Country'].isin(ii)][['Country'] + features],
           '2018': df18[df18['Country'].isin(ii)][['Country'] + features],
           '2019': df19[df19['Country'].isin(ii)][['Country'] + features]}


    dict_economy = {}
    dict_family = {}
    dict_health = {}
    for values in dfs.values():
        for _, row in values.iterrows():
            country = row['Country']
            if country in dict_economy:
                dict_economy[country].append(row.tolist()[1])
                dict_family[country].append(row.tolist()[2])
                dict_health[country].append(row.tolist()[3])
            else:
                dict_economy[country] = [row.tolist()[1]]
                dict_family[country] = [row.tolist()[2]]
                dict_health[country] = [row.tolist()[3]]


    fig5 = go.Figure()

    years = ['2015', '2016', '2017', '2018', '2019']
    colors = ['green', 'green', 'green', 'green', 'green', 'green','rgb(171, 50, 96)','rgb(171, 50, 96)','rgb(171, 50, 96)','rgb(171, 50, 96)','rgb(171, 50, 96)']
    high_countries1 = ['Switzerland', 'Iceland', 'Denmark', 'Norway', 'Canada', 'Finland', 'Togo', 'Burundi', 'Syria', 'Burkina Faso', 'Afghanistan']

    categories = ['Economy', 'Family', 'Health']
    selected_features = st.selectbox('Select features to display:', categories)
    dict_feature = dict_economy
    if selected_features == 'Economy':
        dict_feature = dict_economy
    elif selected_features == 'Family':
        dict_feature = dict_family
    elif selected_features == 'Health':
        dict_feature = dict_health

    for i, country in enumerate(high_countries1):
        fig5.add_trace(go.Scatter(x=years, y=dict_feature[country], name=high_countries1[i], line_width=2.0, line=dict(color=colors[i])))


    fig5.update_layout(title=f'{selected_features} through the years by contry',
                      xaxis_title='Value',
                      yaxis_title='Year',
                      titlefont={'size': 28, 'family':'Serif'},
                      showlegend=True,
                      paper_bgcolor='lightgray',
                      width=750, height=500,
                     )
    st.plotly_chart(fig5)



#####################################################################################################


    
    st.header("Feature Correlation With Happiness Score")
    years = ['2015', '2016', '2017', '2018', '2019']
    selected_year_3 = st.selectbox("Select Year 3",  years)
    df3 = df15

    if selected_year_3 == '2015':
        df3 = df15
    elif selected_year_3 == '2016':
        df3 = df16
    elif selected_year_3 == '2017':
        df3 = df17
    elif selected_year_3 == '2018':
        df3 = df18
    elif selected_year_3 == '2019':
        df3 = df19


    features = ['Economy', 'Generosity', 'Freedom', 'Family', 'Trust', 'Health']

    # Filter the correlation table based on selected features
    correlation_table = df3.corr()[['Score']]
    correlation_table = correlation_table.loc[features].sort_values(by='Score', ascending=False)

    # Taking the absolute values of the correlation values
    correlation_table['Score'] = correlation_table['Score'].abs()
    correlation_table['Score'] = pd.to_numeric(correlation_table['Score'])
    # Create a pie chart using Plotly Express
    fig4 = px.pie(correlation_table, values='Score', names=correlation_table.index)

    # Customize the chart colors
    custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
    fig4.update_traces(marker=dict(colors=custom_colors))

    # Display the chart
    st.plotly_chart(fig4)
###################################################################################
    st.header('3 most influential features through the years by contry')
    st.write('Here you can see the difference between the impact of the fetures through the years by the contries that rank highest and lowest ')

    features = ['Economy', 'Family', 'Health']
    ii = ['Switzerland', 'Iceland', 'Denmark', 'Norway', 'Canada', 'Finland', 'Togo', 'Burundi', 'Syria', 'Burkina Faso', 'Afghanistan']

    dfs = {'2015': df15[df15['Country'].isin(ii)][['Country'] + features],
           '2016': df16[df16['Country'].isin(ii)][['Country'] + features],
           '2017': df17[df17['Country'].isin(ii)][['Country'] + features],
           '2018': df18[df18['Country'].isin(ii)][['Country'] + features],
           '2019': df19[df19['Country'].isin(ii)][['Country'] + features]}


    dict_economy = {}
    dict_family = {}
    dict_health = {}
    for values in dfs.values():
        for _, row in values.iterrows():
            country = row['Country']
            if country in dict_economy:
                dict_economy[country].append(row.tolist()[1])
                dict_family[country].append(row.tolist()[2])
                dict_health[country].append(row.tolist()[3])
            else:
                dict_economy[country] = [row.tolist()[1]]
                dict_family[country] = [row.tolist()[2]]
                dict_health[country] = [row.tolist()[3]]


    fig5 = go.Figure()

    years = ['2015', '2016', '2017', '2018', '2019']
    colors = ['green', 'green', 'green', 'green', 'green', 'green','rgb(171, 50, 96)','rgb(171, 50, 96)','rgb(171, 50, 96)','rgb(171, 50, 96)','rgb(171, 50, 96)']
    high_countries1 = ['Switzerland', 'Iceland', 'Denmark', 'Norway', 'Canada', 'Finland', 'Togo', 'Burundi', 'Syria', 'Burkina Faso', 'Afghanistan']

    categories = ['Economy', 'Family', 'Health']
    selected_features = st.selectbox('Select features to display:', categories)
    dict_feature = dict_economy
    if selected_features == 'Economy':
        dict_feature = dict_economy
    elif selected_features == 'Family':
        dict_feature = dict_family
    elif selected_features == 'Health':
        dict_feature = dict_health

    for i, country in enumerate(high_countries1):
        fig5.add_trace(go.Scatter(x=years, y=dict_feature[country], name=high_countries1[i], line_width=2.0, line=dict(color=colors[i])))


    fig5.update_layout(title=f'{selected_features} through the years by contry',
                      xaxis_title='Value',
                      yaxis_title='Year',
                      titlefont={'size': 28, 'family':'Serif'},
                      showlegend=True,
                      paper_bgcolor='lightgray',
                      width=750, height=500,
                     )
    st.plotly_chart(fig5)

###############################################################################

# # Display the resulting dataframe with the reordered rows
#
# # Create an empty dataframe to store the results
# economy_df = pd.DataFrame(columns=['Country'] + list(dfs.keys()))
#
# # Iterate over the dictionary and extract the Economy values for each year
# for year, df in dfs.items():
#     df['Country'] = pd.Categorical(filtered_df['Country'], categories=ii, ordered=True)
#     economy_df = filtered_df.sort_values('Country')
#     economy_df = economy_df.reset_index(drop=True)
#     economy_df[year] = df['Economy']
#
# # Set the 'Country' column as the index
# economy_df.set_index('Country', inplace=True)
#
# # Display the resulting dataframe
#
# Switzerland_Economy = [1.39651, 1.52733, 1.564980, 1.420, 1.452 ]
# Iceland_Economy = [1.30232, 1.42666, 1.480633, 1.343, 1.380]
# Denmark_Economy = [1.32548, 1.44178, 1.482383, 1.351, 1.383]
# Norway_Economy = [1.45900, 1.57744, 1.616463, 1.456, 1.488]
# Canada_Economy = [1.32629, 1.44015, 1.479204, 1.330, 1.365]
# Finland_Economy = [1.29025,  1.40598, 1.443572, 1.305, 1.340 ]
# names1 = [Switzerland_Economy, Iceland_Economy, Denmark_Economy, Norway_Economy, Canada_Economy, Finland_Economy]
# Togo_F = [0.13995, 0.00000, 0.431883, 0.474, 0.572]
# Burundi_F = [0.41587, 0.23442, 0.629794, 0.627, 1.056, 0.447]
# Syria_F = [0.47489, 0.14866, 0.396103, 0.382, 0.378]
# Burkina_Faso_F = [0.85188, 0.63054, 1.043280, 1.097]
# Afghanistan_F = [0.30285, 0.11037, 0.581543, 0.537, 0.517]
# names2 = [Togo_F, Burundi_F, Syria_F, Burkina_Faso_F, Afghanistan_F]
# namess = [Switzerland_Economy, Iceland_Economy, Denmark_Economy, Norway_Economy, Canada_Economy, Finland_Economy,Togo_F, Burundi_F, Syria_F, Burkina_Faso_F, Afghanistan_F ]

# Switzerland_Family = [1.34951]
    # Iceland_Family = [1.40223]
    # Denmark_Family = [1.36058]
    # Norway_Family = [1.45900]
    # Canada_Family = [1.32629]
    # Finland_Family = [1.29025]
    # Switzerland_Health = [1.39651]
    # Iceland_Health = [1.30232]
    # Denmark_Health = [1.32548]
    # Norway_Health = [1.45900]
    # Canada_Health = [1.32629]
    # Finland_Health = [1.29025]


###################################################################3

    # # Filter data based on the selected years
    # df1 = df[df['year'] == 2015]
    # df2 = df[df['year'] == 2016]
    #
    # # Calculate the sum of each feature for the selected years
    #
    # sum1 = df15['Economy'].sum()
    # sum2 = df16['Economy'].sum()
    #
    # # Calculate the difference between the sums
    # diff = abs(sum1 - sum2)
    # print(diff)
    # diff = diff.values.tolist()
    #
    #
    # # Plotting the donut chart
    # fig, ax = plt.subplots()
    # ax.pie(diff, labels=features, autopct='%1.1f%%', startangle=90)
    # ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
    # plt.title("Impact of Features on Rank")
    # plt.legend(features, loc="center right")
    #
    # # Display the chart
    # st.pyplot(fig)

    #
    # Filter for each hotel type


    # # Display the correlation table
    # correlation_table = df15.corr()[['Rank']].sort_values(by='Rank', ascending=False)
    # # st.write("Correlation with Happiness Score:")
    # # st.dataframe(correlation_table)
    #
    # # Display the correlation heatmap
    # plt.figure(figsize=(5, 8))
    # heatmap = sns.heatmap(correlation_table, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    # heatmap.set_title('Features Correlating with Happiness Score', fontdict={'fontsize': 12}, pad=8)
    # st.pyplot(plt)


    # features = ['Economy', 'Generosity', 'Freedom', 'Family', 'Trust', 'Health']
    #
    # # Filter the correlation table based on selected features
    # correlation_table = df3.corr()[['Score']].loc[features].sort_values(by='Score', ascending=False)
    #
    # # Taking the absolute values of the correlation values
    # correlation_table['Score'] = correlation_table['Score'].abs()
    # custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
    #
    #
    # # Plotting the pie chart
    # fig, ax = plt.subplots()
    # ax.pie(correlation_table['Score'], labels=correlation_table.index, autopct='%1.1f%%', startangle=70, colors=custom_colors)
    # ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle
    # plt.title('Features Correlating with Happiness Score')
    # plt.legend(correlation_table.index, loc='best')

    # Display the chart
    # st.pyplot(fig)
##$$$$
    # df_resort = df_filtered[df_filtered['hotel'] == 'Resort Hotel']
    # df_city = df_filtered[df_filtered['hotel'] == 'City Hotel']
    #
    # # Calculate the total guests for each month
    # resort_guests = df_resort['arrival_month'].value_counts()
    # city_guests = df_city['arrival_month'].value_counts()

    # Pie chart for Resort

################################################################################################




####################################################

    high_countries = ['Switzerland','Iceland','Denmark', 'Norway', 'Canada', 'Finland', 'Netherlands','Israel' ]
    low_country = ['Togo','Burundi','Syria' ,'Syria' ,'Afghanistan', 'Burkina Faso', 'Ivory Coast']

    countries = ['Switzerland', 'Iceland', 'Denmark', 'Norway', 'Canada', 'Finland', 'Netherlands', 'Israel']
    features = ['Economy', 'Family', 'Health']
    #
    # # Filter the DataFrame based on the selected countries
    # data = df15[df15['Country'].isin(countries)][['Country'] + features]
    #
    #
    # years = ['2015', '2016', '2017', '2018', '2019']
    # x = [1,2,6,5,3]
    #
    # # Filter the DataFrame based on the selected countries and years
    # data = df[df['Country'].isin(countries) & df['year'].isin(years)][['Country', 'year', 'Economy']]
    #
    # # Visualization
    # f, ax1 = plt.subplots(figsize=(20, 10))
    # sns.pointplot( data=data, color='lime')
    # sns.pointplot( data=data, color='red')
    # # sns.lineplot(x='year', y='Economy', hue='Country', data=data)
    # plt.xlabel('Year', fontsize=15, color='blue')
    # plt.ylabel('Economy', fontsize=15, color='blue')
    # plt.title('Economy by Year for Selected Countries', fontsize=20, color='blue')
    # plt.legend(title='Country', loc='upper left')
    # plt.grid()
    # st.pyplot(plt)

    # Extract the desired features from the filtered DataFrame


    # Visualization
    # f, ax1 = plt.subplots(figsize=(20, 10))
    # sns.pointplot( data=data, color='lime')
    # sns.pointplot( data=data, color='red')
    # plt.text(7.55, 0.6, 'happiness score ratio', color='red', fontsize=17, style='italic')
    # plt.text(7.55, 0.55, 'economy ratio', color='lime', fontsize=18, style='italic')
    # plt.xticks(rotation=45)
    # plt.xlabel('Region', fontsize=15, color='blue')
    # plt.ylabel('Values', fontsize=15, color='blue')
    # plt.title('Happiness Score  VS  Economy Rate', fontsize=20, color='blue')
    # plt.grid()
    # st.pyplot(plt)

#########################################################



    # st.subheader("Rank Table")

    # df11 = pd.merge(df1[['Country', 'Rank']], df2[['Country', 'Rank']], on='Country')
    # df11.columns = ['Country', 'Year 1', 'Year 2']
    # df11['Rank Change'] = df11['Year 1'] - df11['Year 2']



    # st.subheader("Rank Table")
    #
    # # Create a selectbox with a default value of "None"
    #
    #
    # if selected_country is not None:
    #     # Perform the desired action when a country is selected
    #     st.write("You selected:", selected_country)
    # else:
    #     # Handle the case when "None" is selected
    #     st.write("No country selected.")



    # Filter the DataFrame based on the selected country

##########################################################################################################


#############################  Features importance comparison  ######################################################
    # st.title("Features importance comparison")
    # st.write("Choose another country, year and features and comare the difference")
    # selected_country2 = st.selectbox("Select second country for comparsion", all_country)
    # selected_year3 = st.selectbox("Select a year", years)
    # dff = df15
    # # Define the corresponding DataFrame based on the selected year
    # if selected_year3 == '2015':
    #     dff = df15
    # elif selected_year3 == '2016':
    #     dff = df16
    # elif selected_year3 == '2017':
    #     dff = df17
    # elif selected_year3 == '2018':
    #     dff = df18
    # elif selected_year3 == '2019':
    #     dff = df19
    # categories = ['Score', 'Economy', 'Generosity', 'Freedom', 'Family', 'Trust', 'Health']
    # selected_features = st.multiselect('Select features to display:', categories)
    #
    # r1 = [dff[each][dff["Country"] == selected_country].mean() / dff[each].max() for each in selected_features]
    # r2 = [dff[each][dff["Country"] == selected_country2].mean() / dff[each].max() for each in selected_features]
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatterpolar(
    #     r=r1,
    #     theta=categories,
    #     fill='toself',
    #     name=selected_country
    # ))
    # fig.add_trace(go.Scatterpolar(
    #     r=r2,
    #     theta=categories,
    #     fill='toself',
    #     name=selected_country2
    # ))
    #
    # fig.update_layout(
    #     polar=dict(
    #         radialaxis=dict(
    #             visible=True,
    #             range=[0, 1]
    #         )),
    #     showlegend=False
    # )
    # st.plotly_chart(fig)


################################## bar plot  #####################################################

    # for feature in selected_features:
    #     data = df[df["Country"] == selected_country][feature].tolist()
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.bar(years, data, color='maroon', width=0.4)
    #     plt.xlabel("Year")
    #     plt.ylabel(feature)
    #     plt.title(f"{feature} in {selected_country} by Year")
    #     st.pyplot(fig)



########################  World map  #######################################################




# Assuming you have your own dataframe named 'df' with the required columns







    # st.title("World map")
    # # Create the interactive plot
    # fig1 = px.choropleth(df15, Country = 'Country', color='total',
    #                      title='World map',
    #                      labels={'total': 'Score', 'Country': 'Country'},
    #                      color_continuous_scale=px.colors.sequential.Plasma,
    #                      projection='natural earth',
    #                      hover_data={'Score': True, 'Country': True})
    #
    # fig1
    #








#########################################################################3
    # st.subheader("Heatmap 2019")
    # fig2019, ax2019 = plt.subplots()
    # sns.heatmap(df19.iloc[:, 2:].corr(), annot=True, cmap='coolwarm', ax=ax2019)
    # st.pyplot(fig2019)






###################################################################################
    # st.title("Features correlation by year")
    # st.write("Which feature influences Happiness the most?")
    #
    #
    # # Create a search box for the year
    # selected_year = st.selectbox("Select a year", ['2015', '2016', '2017', '2018', '2019'])
    # df4 = df19
    # # Define the corresponding DataFrame based on the selected year
    # if selected_year == '2015':
    #     df4 = df15
    # elif selected_year == '2016':
    #     df4 = df16
    # elif selected_year == '2017':
    #     df4 = df17
    # elif selected_year == '2018':
    #     df4 = df18
    #
    # # Generate the heatmap for the selected year
    # fig, ax = plt.subplots()
    # sns.heatmap(df4.iloc[:, 2:].corr(), annot=True, cmap='coolwarm', ax=ax)
    # st.pyplot(fig)
# ###############################################################################
    # score_data = pd.concat(
    #     [df15["Country"], df15["Score"], df16["Score"], df17["Score"],
    #      df18["Score"], df19["Score"]], axis=1)
    # score_data["Score_difference"] = 0
    # scoresGroupByCountry = score_data.groupby(["Country", "Score_difference"]).sum().reset_index()
    # country = list(scoresGroupByCountry["Country"])
    # diff = list(scoresGroupByCountry["Score_difference"])
    # list1 = [[country[i], diff[i]] for i in range(scoresGroupByCountry.shape[0])]
    # map_1 = Map(init_opts=opts.InitOpts(width='1000px', height='460px', theme=ThemeType.ROMANTIC))
    # map_1.add('Rank Difference',
    #           list1,
    #           maptype='world',
    #           is_map_symbol_show=False)
    # map_1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    # map_1.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=611, is_piecewise=True, pieces=[
    #     {"max": -3},
    #     {"max": -2, "max": -1},
    #     {"max": -1, "max": 0},
    #     {"max": 0, "max": 1},
    #     {"max": 1, "max": 2},
    #     {"max": 2, "max": 3}]),
    #                       title_opts=opts.TitleOpts(
    #                           title='Happiness Score Difference By Countries',
    #                           pos_left='center',
    #                           padding=0,
    #                           item_gap=2,
    #                           title_textstyle_opts=opts.TextStyleOpts(color='Black',
    #                                                                   font_weight='bold',
    #                                                                   font_family='Courier New',
    #                                                                   font_size=30),
    #                           subtitle_textstyle_opts=opts.TextStyleOpts(color='grey',
    #                                                                      font_weight='bold',
    #                                                                      font_family='Courier New',
    #                                                                      font_size=13)),
    #                       legend_opts=opts.LegendOpts(is_show=False))
    # # st.pyplot(map_1)
    # # st.pydeck_chart(map_1.to_json())
    #
    #
    # # Display the HTML file in Streamlit
    # map_html = map_1.render_notebook()
    #
    # with open("map_1.html", "r") as f:
    #     map_html = f.read()
    # st.components.v1.html(map_html, width=1000, height=460)
    #
    # import geopandas as gpd
    # import pydeck as pdk
    #
    # # pydeck_data = data_utils.df_to_json(list1, lat="latitude", lon="longitude")
    # df = pd.DataFrame(list1, columns=["Country", "Score_difference"])
    # # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    #
    # # Convert the GeoDataFrame to GeoJSON
    # geojson_data = df.to_json()
    #
    # # Configure the PyDeck visualization
    # pydeck_config = {
    #     "mapStyle": "mapbox://styles/mapbox/light-v9",
    #     "layers": [
    #         {
    #             "type": "ScatterplotLayer",
    #             "data": geojson_data,
    #             "radiusScale": 10000,
    #             "radiusMinPixels": 2,
    #             "getFillColor": [255, 0, 0],
    #             "pickable": True,
    #         }
    #     ],
    # }
    #
    # # Render the PyDeck visualization using Streamlit
    # st.pydeck_chart(pdk.Deck(map_1))




    # map_1.render_notebook()
    # data = dict(type='choropleth',
    #             locations=df17["Country"],
    #             locationmode='country names',
    #             z=df17["Score"],
    #             text=df17["Country"],
    #             colorbar={"title": "Score"})
    #
    # layout = dict(title="Geographical Visualization of Happiness Score",
    #               geo=dict(showframe=True, projection={"type": "azimuthal equal area"}))
    #
    # happiness_map = go.Figure(data=[data], layout=layout)
    #
    # # Convert the figure to a dictionary
    # fig_dict = happiness_map.to_dict()
    #
    # # Print the plot
    # st.pyplot(fig_dict)


    # fig1 = px.choropleth(df, locations='Country', color='Country',
    #                     title='Number of Bookings Per Country',
    #                     labels={'total': 'Total Number of Bookings', 'country': 'Country'},
    #                     hover_name='Country',
    #                     color_continuous_scale=px.colors.sequential.Plasma)
    # st.pyplot(fig1)
##################################################################################################################
# st.subheader("Rank Table")

# df11 = pd.merge(df1[['Country', 'Rank']], df2[['Country', 'Rank']], on='Country')
# df11.columns = ['Country', 'Year 1', 'Year 2']
# df11['Rank Change'] = df11['Year 1'] - df11['Year 2']


# st.subheader("Rank Table")
#
# # Create a selectbox with a default value of "None"
#
#
# if selected_country is not None:
#     # Perform the desired action when a country is selected
#     st.write("You selected:", selected_country)
# else:
#     # Handle the case when "None" is selected
#     st.write("No country selected.")


# Filter the DataFrame based on the selected country
##################################################################################################################
# Run the app
app()
